"""Real-data and CUDA smoke gates for formal E0/E1 training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml

from scripts.train_e0 import (
    _first_training_batch,
    _seed_everything,
    build_train_validation_datasets,
    load_formal_config,
    select_physical_batch_size,
)
from src.models.tsvit_segmentation import TSViTSegmentation
from src.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from src.training.evaluate import evaluate_tiles
from src.training.schedule import WarmupCosineSchedule


class _SingleTileDataset:
    def __init__(self, dataset, index: int = 0) -> None:
        self.dataset = dataset
        self.index = index

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.dataset[self.index]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=ROOT / "configs" / "paths.local.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "xinjiang_2021_e0_e1.csv")
    parser.add_argument("--normalization", type=Path, default=ROOT / "manifests" / "xinjiang_2021_train_normalization.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "models" / "tsvit_baseline.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".smoke" / "formal")
    parser.add_argument("--overfit-steps", type=int, default=8)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = load_formal_config(args.config)
    with args.paths.open("r", encoding="utf-8") as stream:
        workspace_root = Path(yaml.safe_load(stream)["workspace_root"])
    normalization = json.loads(args.normalization.read_text(encoding="utf-8"))
    train_dataset, validation_dataset = build_train_validation_datasets(
        args.manifest, workspace_root, normalization
    )
    train_sample = train_dataset[0]
    validation_sample = validation_dataset[0]
    assert train_sample["images"].shape == (26, 10, 256, 256)
    assert validation_sample["label"].shape == (256, 256)
    assert train_sample["split"] == "train"
    assert validation_sample["split"] == "validation"
    assert torch.isfinite(train_sample["images"][train_sample["valid_mask"]]).all()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal E0/E1 real-data smoke requires CUDA")
    _seed_everything(int(config["experiment"]["seed"]))
    torch.cuda.reset_peak_memory_stats(device)
    model = TSViTSegmentation(config["model"]).to(device)
    data = config["data"]
    training = config["training"]
    physical_batch = select_physical_batch_size(
        model,
        train_dataset,
        training["physical_batch_candidates"],
        device=device,
        window_size=data["window_size"],
        stride=data["stride"],
        amp=training["amp"],
    )
    smoke_batch = min(2, physical_batch)
    images, labels, doy, valid_mask, sample_id = _first_training_batch(
        train_dataset, smoke_batch, data["window_size"], data["stride"]
    )
    images = images.to(device)
    labels = labels.to(device)
    doy = doy.to(device)
    valid_mask = valid_mask.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = WarmupCosineSchedule(optimizer, args.overfit_steps, 0, 1e-3, 1e-3, 1e-3)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(training["amp"]),
        init_scale=float(training["amp_init_scale"]),
        growth_interval=int(training["amp_growth_interval"]),
    )
    losses: list[float] = []
    model.train()
    for _ in range(args.overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images, doy, valid_mask)
            loss = F.cross_entropy(logits, labels, ignore_index=data["ignore_index"])
        if not torch.isfinite(loss):
            raise FloatingPointError("real-batch overfit loss is non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError("real-batch overfit gradient is non-finite")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(float(loss.detach().item()))
    if losses[-1] >= losses[0]:
        raise AssertionError(f"fixed real-batch loss did not decrease: {losses}")

    validation_metrics = evaluate_tiles(
        model,
        _SingleTileDataset(validation_dataset),
        device=device,
        window_size=data["window_size"],
        stride=data["stride"],
        window_batch_size=physical_batch,
        ignore_index=data["ignore_index"],
        amp=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "reload_probe.pt"
    amp_policy = {
        "init_scale": float(training["amp_init_scale"]),
        "backoff_factor": float(training["amp_backoff_factor"]),
        "min_scale": float(training["amp_min_scale"]),
        "max_backoffs_per_batch": int(training["amp_max_backoffs_per_batch"]),
        "growth_interval": int(training["amp_growth_interval"]),
    }
    run_config = {
        "smoke": config["experiment"]["id"],
        "physical_batch_size": physical_batch,
        "amp_policy": amp_policy,
    }
    state = CheckpointState(epoch=1, global_step=args.overfit_steps, best_metric=0.0)
    save_checkpoint(checkpoint, model, optimizer, scheduler, scaler, state, run_config)
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        expected_logits = model(images, doy, valid_mask).float()
    restored_model = TSViTSegmentation(config["model"]).to(device)
    restored_optimizer = torch.optim.AdamW(
        restored_model.parameters(), lr=1e-3, weight_decay=0.0
    )
    restored_scheduler = WarmupCosineSchedule(
        restored_optimizer, args.overfit_steps, 0, 1e-3, 1e-3, 1e-3
    )
    restored_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(training["amp"]),
        init_scale=float(training["amp_init_scale"]),
        growth_interval=int(training["amp_growth_interval"]),
    )
    restored_state = load_checkpoint(
        checkpoint,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_scaler,
        expected_run_config=run_config,
        map_location=device,
    )
    restored_model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        restored_logits = restored_model(images, doy, valid_mask).float()
    torch.testing.assert_close(restored_logits, expected_logits, rtol=0.0, atol=0.0)
    if restored_state != state:
        raise AssertionError("checkpoint training state did not round-trip")

    result = {
        "status": "passed",
        "experiment_id": config["experiment"]["id"],
        "device": torch.cuda.get_device_name(0),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "amp_policy": amp_policy,
        "physical_batch_size": physical_batch,
        "effective_batch_size": training["effective_batch_size"],
        "sample_id": sample_id,
        "overfit_losses": losses,
        "single_tile_validation": validation_metrics,
        "checkpoint_reload_exact": True,
        "test_records_loaded": 0,
    }
    (args.output_dir / "smoke_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
