"""Train or replay Validation for formal Xinjiang 2021 E0/E1 experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.data.manifest import TileRecord, load_manifest, sha256_file
from src.data.windows import extract_windows, window_coordinates
from src.data.xinjiang_tiles import XinjiangTileDataset
from src.models.tsvit_segmentation import TSViTSegmentation
from src.training.checkpoint import CheckpointState, load_checkpoint
from src.training.evaluate import evaluate_tiles
from src.training.schedule import WarmupCosineSchedule
from src.training.trainer import E0Trainer, TrainerConfig


FORMAL_OUTPUTS = {
    "E0": Path(
        r"E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e0_tsvit_doy_seed42"
    ),
    "E1": Path(
        r"E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e1_tsvit_doy_wpe_seed42"
    ),
}


def formal_output_for(config: Mapping[str, Any]) -> Path:
    experiment_id = str(config["experiment"]["id"])
    try:
        return FORMAL_OUTPUTS[experiment_id]
    except KeyError as error:
        raise ValueError(f"unsupported formal experiment {experiment_id}") from error


def resolve_output_dir(
    config: Mapping[str, Any], requested: Path | None
) -> Path:
    return requested or formal_output_for(config)


def load_formal_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"experiment", "model", "data", "training", "evaluation"}
    if not isinstance(config, dict) or not required.issubset(config):
        raise ValueError(f"formal configuration requires sections {sorted(required)}")
    experiment_id = str(config["experiment"]["id"])
    if experiment_id not in FORMAL_OUTPUTS:
        raise ValueError("formal training requires experiment.id in {E0,E1}")
    if config["model"]["image_size"] != 24 or config["model"]["patch_size"] != 2:
        raise ValueError("formal E0/E1 requires image_size=24 and patch_size=2")
    wavelet = config["model"].get("wavelet", {})
    if experiment_id == "E0" and bool(wavelet.get("enabled", False)):
        raise ValueError("formal E0 requires WPE disabled")
    expected_e1_wavelet = {
        "enabled": True,
        "scale_init_days": [7.0, 17.5, 35.0],
        "scale_min_days": 3.5,
        "scale_max_days": 35.0,
        "shift_init_days": [0.0, 0.0, 0.0],
        "shift_max_abs_days": 7.0,
        "support_radius_days": 42.0,
        "alpha_init": 0.01,
        "eps": 1.0e-6,
    }
    if experiment_id == "E1" and wavelet != expected_e1_wavelet:
        if not bool(wavelet.get("enabled", False)):
            raise ValueError("formal E1 requires WPE enabled")
        raise ValueError("formal E1 WPE parameters differ from the frozen design")
    expected_training = {
        "max_epochs": 100,
        "effective_batch_size": 16,
        "physical_batch_candidates": [16, 8, 4, 2],
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "start_lr": 1e-8,
        "base_lr": 1e-3,
        "min_lr": 5e-6,
        "warmup_epochs": 10,
        "early_stopping_patience": 12,
        "amp": True,
        "amp_init_scale": 8192.0,
        "amp_backoff_factor": 0.5,
        "amp_min_scale": 128.0,
        "amp_max_backoffs_per_batch": 6,
        "amp_growth_interval": 1_000_000,
    }
    if config["training"] != expected_training:
        raise ValueError("formal E0/E1 training parameters differ from the frozen design")
    if config["evaluation"] != {
        "selection_metric": "maize_iou",
        "validation_every_epochs": 1,
        "test_during_training": False,
    }:
        raise ValueError("formal E0/E1 evaluation policy differs from the frozen design")
    return config


def build_train_validation_datasets(
    manifest_path: Path,
    workspace_root: Path,
    normalization: dict[str, Any],
    *,
    dataset_factory: Callable[..., Any] = XinjiangTileDataset,
    expected_counts: tuple[int, int] | None = (495, 276),
) -> tuple[Any, Any]:
    records = load_manifest(
        manifest_path,
        workspace_root,
        selected_splits={"train", "validation"},
    )
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    if expected_counts is not None and (len(train_records), len(validation_records)) != expected_counts:
        raise ValueError(
            "formal split counts must be "
            f"train={expected_counts[0]}, validation={expected_counts[1]}; got "
            f"{len(train_records)}, {len(validation_records)}"
        )
    return (
        dataset_factory(train_records, workspace_root, normalization=normalization),
        dataset_factory(validation_records, workspace_root, normalization=normalization),
    )


def optimizer_steps_per_epoch(
    label_paths: Iterable[Path],
    *,
    window_size: int,
    stride: int,
    effective_batch: int,
    ignore_index: int = 255,
) -> int:
    if effective_batch <= 0:
        raise ValueError("effective_batch must be positive")
    steps = 0
    for path in label_paths:
        label = np.load(Path(path), allow_pickle=False)
        coordinates = window_coordinates(
            int(label.shape[0]), int(label.shape[1]), window_size, stride
        )
        valid_windows = sum(
            not bool(
                np.all(
                    label[row : row + window_size, column : column + window_size]
                    == ignore_index
                )
            )
            for row, column in coordinates
        )
        steps += math.ceil(valid_windows / effective_batch)
    if steps <= 0:
        raise ValueError("Train labels contain no supervised windows")
    return steps


def _first_training_batch(dataset: Any, size: int, window_size: int, stride: int):
    for index in range(len(dataset)):
        sample = dataset[index]
        coordinates = window_coordinates(256, 256, window_size, stride)
        images, labels, kept = extract_windows(
            sample["images"],
            sample["label"],
            coordinates,
            window_size=window_size,
            skip_all_ignore=True,
        )
        if len(kept) >= size:
            return (
                images[:size],
                labels[:size],
                sample["doy"].unsqueeze(0).expand(size, -1),
                sample["valid_mask"].unsqueeze(0).expand(size, -1),
                sample["sample_id"],
            )
    raise ValueError(f"could not find {size} supervised Train windows")


def select_physical_batch_size(
    model: torch.nn.Module,
    dataset: Any,
    candidates: Sequence[int],
    *,
    device: torch.device,
    window_size: int,
    stride: int,
    amp: bool,
) -> int:
    original = copy.deepcopy(model.state_dict())
    model.to(device).train()
    last_error: RuntimeError | None = None
    for candidate in candidates:
        try:
            images, labels, doy, valid_mask, _ = _first_training_batch(
                dataset, int(candidate), window_size, stride
            )
            model.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                logits = model(
                    images.to(device), doy.to(device), valid_mask.to(device)
                )
                loss = F.cross_entropy(logits, labels.to(device), ignore_index=255)
            loss.backward()
            if not torch.isfinite(loss) or any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("batch probe produced non-finite values")
            model.zero_grad(set_to_none=True)
            model.load_state_dict(original)
            return int(candidate)
        except torch.OutOfMemoryError as error:
            last_error = error
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    model.load_state_dict(original)
    raise RuntimeError("no approved physical batch size fits GPU memory") from last_error


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--porcelain=v1").splitlines(),
    }


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=ROOT / "configs" / "paths.local.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "xinjiang_2021_e0_e1.csv")
    parser.add_argument("--normalization", type=Path, default=ROOT / "manifests" / "xinjiang_2021_train_normalization.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "models" / "tsvit_baseline.yaml")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--physical-batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-only", type=Path, default=None, metavar="CHECKPOINT")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_formal_config(args.config)
    output_dir = resolve_output_dir(config, args.output_dir)
    with args.paths.open("r", encoding="utf-8") as stream:
        workspace_root = Path(yaml.safe_load(stream)["workspace_root"])
    with args.normalization.open("r", encoding="utf-8") as stream:
        normalization = json.load(stream)
    train_dataset, validation_dataset = build_train_validation_datasets(
        args.manifest, workspace_root, normalization
    )
    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal E0/E1 training requires CUDA")
    model = TSViTSegmentation(config["model"])
    data_config = config["data"]
    training = config["training"]
    max_epochs = int(args.max_epochs or training["max_epochs"])
    if max_epochs <= 0 or max_epochs > training["max_epochs"]:
        raise ValueError("max_epochs must lie in [1,100]")
    physical_batch = args.physical_batch_size
    if physical_batch is None:
        physical_batch = select_physical_batch_size(
            model,
            train_dataset,
            training["physical_batch_candidates"],
            device=device,
            window_size=data_config["window_size"],
            stride=data_config["stride"],
            amp=training["amp"],
        )
    if physical_batch not in training["physical_batch_candidates"]:
        raise ValueError("physical_batch_size must be one of the approved candidates")
    label_paths = [
        record.resolve(workspace_root).label_path for record in train_dataset.records
    ]
    steps_per_epoch = optimizer_steps_per_epoch(
        label_paths,
        window_size=data_config["window_size"],
        stride=data_config["stride"],
        effective_batch=training["effective_batch_size"],
        ignore_index=data_config["ignore_index"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["base_lr"], weight_decay=training["weight_decay"]
    )
    scheduler = WarmupCosineSchedule(
        optimizer,
        total_steps=steps_per_epoch * max_epochs,
        warmup_steps=steps_per_epoch * min(training["warmup_epochs"], max_epochs),
        start_lr=training["start_lr"],
        base_lr=training["base_lr"],
        min_lr=training["min_lr"],
    )
    run_config = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "model": config["model"],
        "data": data_config,
        "training": {**training, "max_epochs": max_epochs},
        "manifest_sha256": sha256_file(args.manifest),
        "normalization_sha256": sha256_file(args.normalization),
        "physical_batch_size": physical_batch,
        "steps_per_epoch": steps_per_epoch,
        "seed": seed,
    }
    trainer_config = TrainerConfig(
        max_epochs=max_epochs,
        window_size=data_config["window_size"],
        stride=data_config["stride"],
        physical_batch_size=physical_batch,
        effective_batch_size=training["effective_batch_size"],
        ignore_index=data_config["ignore_index"],
        amp=training["amp"],
        amp_init_scale=training["amp_init_scale"],
        amp_backoff_factor=training["amp_backoff_factor"],
        amp_min_scale=training["amp_min_scale"],
        amp_max_backoffs_per_batch=training["amp_max_backoffs_per_batch"],
        amp_growth_interval=training["amp_growth_interval"],
        seed=seed,
        warmup_epochs=training["warmup_epochs"],
        patience=training["early_stopping_patience"],
        validation_batch_size=physical_batch,
    )
    trainer = E0Trainer(
        model,
        train_dataset,
        validation_dataset,
        optimizer,
        scheduler,
        output_dir=output_dir,
        device=device,
        config=trainer_config,
        run_config=run_config,
    )
    snapshot = {
        "run_config": run_config,
        "workspace_root": str(workspace_root.resolve()),
        "config_path": str(args.config.resolve()),
        "manifest_path": str(args.manifest.resolve()),
        "normalization_path": str(args.normalization.resolve()),
        "git": _git_snapshot(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "command": [sys.executable, *sys.argv],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    initial_state = CheckpointState()
    checkpoint = args.validation_only or (output_dir / "last.pt")
    if args.validation_only is not None or (args.resume and checkpoint.is_file()):
        initial_state = load_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            trainer.scaler,
            expected_run_config=run_config,
            map_location=device,
        )
    if args.validation_only is not None:
        metrics = evaluate_tiles(
            model,
            validation_dataset,
            device=device,
            window_size=data_config["window_size"],
            stride=data_config["stride"],
            window_batch_size=physical_batch,
            ignore_index=data_config["ignore_index"],
            amp=training["amp"],
        )
        output = output_dir / "validation_replay.json"
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, sort_keys=True), flush=True)
        return
    final_state = trainer.fit(initial_state)
    print(json.dumps({"terminal_state": final_state.__dict__}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
