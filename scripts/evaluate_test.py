"""Strict one-time Test evaluation for frozen formal E0/E1 checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch import nn

from scripts.train_e0 import _git_snapshot, _seed_everything, load_formal_config
from src.data.manifest import load_manifest, sha256_file
from src.data.xinjiang_tiles import XinjiangTileDataset
from src.models.tsvit_segmentation import TSViTSegmentation
from src.training.evaluate import evaluate_tiles


FROZEN_MANIFEST_SHA256 = (
    "79DCAAF1270D48B99FA50AF2A57548B50D7FD3E232620D11DC53E5E64C1177A8"
)
FROZEN_NORMALIZATION_SHA256 = (
    "1401DC9AAFE9A30A8916AB1A3C738080DB63D433C04716CA27FA5CABB51EC142"
)


FROZEN_TRAINING_CONFIG_FIELDS = (
    "max_epochs",
    "effective_batch_size",
    "physical_batch_candidates",
    "optimizer",
    "weight_decay",
    "start_lr",
    "base_lr",
    "min_lr",
    "warmup_epochs",
    "early_stopping_patience",
    "amp",
    "amp_init_scale",
    "amp_growth_interval",
)

@dataclass(frozen=True)
class FrozenTestAsset:
    experiment_id: str
    checkpoint: Path
    sha256: str
    epoch: int
    validation_maize_iou: float


FROZEN_TEST_ASSETS = {
    "E0": FrozenTestAsset(
        experiment_id="E0",
        checkpoint=Path(
            "06_models/retrain_outputs/maize_model_research/"
            "e0_tsvit_doy_seed42/best.pt"
        ),
        sha256="CAB74C64897DA7FEA8A1A458ED94DAC2E23C0054A1772EAF16CF7BB5C3F9DE86",
        epoch=13,
        validation_maize_iou=0.9383473661376481,
    ),
    "E1": FrozenTestAsset(
        experiment_id="E1",
        checkpoint=Path(
            "06_models/retrain_outputs/maize_model_research/"
            "e1_tsvit_doy_wpe_seed42/best.pt"
        ),
        sha256="60B7A2C0715D45B20F5723DAD8C5ED8CB33E785AAD2A15AD39797A512E0CDDC8",
        epoch=8,
        validation_maize_iou=0.9335717434114438,
    ),
}


def build_test_dataset(
    manifest_path: Path,
    workspace_root: Path,
    normalization: Mapping[str, Any],
    *,
    dataset_factory: Callable[..., Any] = XinjiangTileDataset,
    expected_count: int = 305,
) -> Any:
    records = load_manifest(
        manifest_path,
        workspace_root,
        allowed_splits={"test"},
        selected_splits={"test"},
    )
    if len(records) != expected_count:
        raise ValueError(
            f"expected {expected_count} Test records; got {len(records)}"
        )
    return dataset_factory(records, workspace_root, normalization=normalization)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"checkpoint {label} does not match the frozen evaluation")


def _require_frozen_training_compatibility(
    run_config: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    run_training = run_config.get("training")
    expected_training = config.get("training")
    if run_training is None and expected_training is None:
        return
    if not isinstance(run_training, Mapping) or not isinstance(
        expected_training, Mapping
    ):
        raise ValueError("checkpoint training configuration is incomplete")
    for field in FROZEN_TRAINING_CONFIG_FIELDS:
        _require_equal(
            run_training.get(field),
            expected_training.get(field),
            f"training configuration {field}",
        )

def load_frozen_checkpoint_for_test(
    checkpoint: Path,
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    manifest_sha256: str,
    normalization_sha256: str,
    map_location: str | torch.device,
    asset: FrozenTestAsset | None = None,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    experiment_id = str(config["experiment"]["id"])
    frozen = asset or FROZEN_TEST_ASSETS.get(experiment_id)
    if frozen is None or frozen.experiment_id != experiment_id:
        raise ValueError(f"no frozen Test asset for experiment {experiment_id}")
    checkpoint = Path(checkpoint)
    expected_checkpoint = frozen.checkpoint
    if not expected_checkpoint.is_absolute():
        if workspace_root is None:
            raise ValueError(
                "workspace_root is required for a relative frozen checkpoint"
            )
        expected_checkpoint = Path(workspace_root) / expected_checkpoint
    if checkpoint.resolve() != expected_checkpoint.resolve():
        raise ValueError("checkpoint path does not match the frozen Test asset")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != frozen.sha256:
        raise ValueError("checkpoint SHA256 does not match the frozen Test asset")

    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    run_config = payload.get("run_config")
    training_state = payload.get("training_state")
    if not isinstance(run_config, dict) or not isinstance(training_state, dict):
        raise ValueError("checkpoint metadata is incomplete")

    _require_equal(run_config.get("experiment"), config["experiment"], "experiment")
    _require_equal(run_config.get("model"), config["model"], "model configuration")
    _require_equal(run_config.get("data"), config["data"], "data configuration")
    _require_frozen_training_compatibility(run_config, config)
    _require_equal(run_config.get("manifest_sha256"), manifest_sha256, "manifest SHA256")
    _require_equal(
        run_config.get("normalization_sha256"),
        normalization_sha256,
        "normalization SHA256",
    )
    _require_equal(run_config.get("physical_batch_size"), 16, "physical batch size")
    _require_equal(run_config.get("steps_per_epoch"), 3327, "steps per epoch")
    _require_equal(run_config.get("seed"), 42, "seed")
    _require_equal(training_state.get("epoch"), frozen.epoch, "best epoch")
    if not math.isclose(
        float(training_state.get("best_metric", float("nan"))),
        frozen.validation_maize_iou,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "checkpoint Validation maize IoU does not match the frozen Test asset"
        )

    model.load_state_dict(payload["model"])
    return {
        "epoch": int(training_state["epoch"]),
        "global_step": int(training_state["global_step"]),
        "validation_maize_iou": float(training_state["best_metric"]),
        "checkpoint_sha256": checkpoint_sha256,
    }


def reserve_test_output_directory(checkpoint: Path) -> Path:
    output = Path(checkpoint).parent / "test_evaluation"
    if output.exists():
        raise FileExistsError(f"test output directory already exists: {output}")
    try:
        output.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"test output directory already exists: {output}"
        ) from error
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths", type=Path, default=ROOT / "configs" / "paths.local.yaml"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "manifests" / "xinjiang_2021_e0_e1.csv",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=ROOT / "manifests" / "xinjiang_2021_train_normalization.json",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args(argv)


def build_test_evaluation_document(
    *,
    experiment_id: str,
    checkpoint: Path,
    checkpoint_metadata: Mapping[str, Any],
    manifest_sha256: str,
    normalization_sha256: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    evaluated_tiles = int(metrics["evaluated_tiles"])
    if evaluated_tiles != 305:
        raise ValueError(f"formal Test evaluation requires 305 tiles; got {evaluated_tiles}")
    confusion_matrix = metrics["confusion_matrix"]
    confusion_pixels = sum(int(value) for row in confusion_matrix for value in row)
    supervised_pixels = int(metrics["supervised_pixels"])
    if confusion_pixels != supervised_pixels:
        raise ValueError(
            "confusion matrix total does not match supervised pixel count"
        )
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "evaluation_split": "test",
        "interpretation": (
            "spatially held-out pseudo-label agreement; "
            "not independent ground-truth accuracy"
        ),
        "checkpoint": {
            "path": str(Path(checkpoint).resolve()),
            "sha256": checkpoint_metadata["checkpoint_sha256"],
            "epoch": int(checkpoint_metadata["epoch"]),
            "global_step": int(checkpoint_metadata["global_step"]),
            "validation_maize_iou": float(
                checkpoint_metadata["validation_maize_iou"]
            ),
        },
        "manifest_sha256": manifest_sha256,
        "normalization_sha256": normalization_sha256,
        "metrics": dict(metrics),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_formal_config(args.config)
    experiment_id = str(config["experiment"]["id"])
    manifest_sha256 = sha256_file(args.manifest)
    normalization_sha256 = sha256_file(args.normalization)
    if manifest_sha256 != FROZEN_MANIFEST_SHA256:
        raise ValueError("manifest SHA256 does not match the frozen Test design")
    if normalization_sha256 != FROZEN_NORMALIZATION_SHA256:
        raise ValueError("normalization SHA256 does not match the frozen Test design")

    with args.paths.open("r", encoding="utf-8") as stream:
        workspace_root = Path(yaml.safe_load(stream)["workspace_root"])
    with args.normalization.open("r", encoding="utf-8") as stream:
        normalization = json.load(stream)

    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal E0/E1 Test evaluation requires CUDA")
    model = TSViTSegmentation(config["model"]).to(device)
    checkpoint_metadata = load_frozen_checkpoint_for_test(
        args.checkpoint,
        model,
        config,
        manifest_sha256=manifest_sha256,
        normalization_sha256=normalization_sha256,
        map_location=device,
        workspace_root=workspace_root,
    )

    output_dir = reserve_test_output_directory(args.checkpoint)
    snapshot = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "evaluation_split": "test",
        "workspace_root": str(workspace_root.resolve()),
        "config_path": str(args.config.resolve()),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "normalization_path": str(args.normalization.resolve()),
        "normalization_sha256": normalization_sha256,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
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
    _write_json_atomic(output_dir / "run_snapshot.json", snapshot)

    test_dataset = build_test_dataset(
        args.manifest,
        workspace_root,
        normalization,
        expected_count=305,
    )
    data_config = config["data"]
    metrics = evaluate_tiles(
        model,
        test_dataset,
        device=device,
        window_size=int(data_config["window_size"]),
        stride=int(data_config["stride"]),
        window_batch_size=16,
        ignore_index=int(data_config["ignore_index"]),
        amp=bool(config["training"]["amp"]),
        expected_split="test",
    )
    document = build_test_evaluation_document(
        experiment_id=experiment_id,
        checkpoint=args.checkpoint,
        checkpoint_metadata=checkpoint_metadata,
        manifest_sha256=manifest_sha256,
        normalization_sha256=normalization_sha256,
        metrics=metrics,
    )
    _write_json_atomic(output_dir / "test_evaluation.json", document)
    print(json.dumps(document, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
