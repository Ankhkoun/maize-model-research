"""One-time frozen Test evaluation against independent 30m annual reference labels."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from scripts.evaluate_test import (
    FROZEN_MANIFEST_SHA256,
    FROZEN_NORMALIZATION_SHA256,
    _write_json_atomic,
    build_test_dataset,
    load_frozen_checkpoint_for_test,
)
from scripts.train_e0 import _git_snapshot, _seed_everything, load_formal_config
from src.data.manifest import sha256_file
from src.models.tsvit_segmentation import TSViTSegmentation
from src.training.ground_truth_evaluate import evaluate_ground_truth_tiles


GROUND_TRUTH_RELATIVE_ROOT = Path(
    "03_processed_data/labels_30m/xinjiang_2021/2021"
)


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
    parser.add_argument("--ground-truth-root", type=Path, default=GROUND_TRUTH_RELATIVE_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args(argv)


def resolve_ground_truth_root(workspace_root: Path, root: Path) -> Path:
    root = Path(root)
    return root if root.is_absolute() else Path(workspace_root) / root


def reserve_ground_truth_output_directory(checkpoint: Path) -> Path:
    output = Path(checkpoint).parent / "test_evaluation_ground_truth"
    if output.exists():
        raise FileExistsError(
            f"ground-truth Test output directory already exists: {output}"
        )
    try:
        (output / "native30m").mkdir(parents=True)
        (output / "upsampled10m").mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"ground-truth Test output directory already exists: {output}"
        ) from error
    return output


def build_ground_truth_evaluation_document(
    *,
    experiment_id: str,
    scale: str,
    checkpoint: Path,
    checkpoint_metadata: Mapping[str, Any],
    manifest_sha256: str,
    normalization_sha256: str,
    ground_truth_root: Path,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if scale not in {"native30m", "upsampled10m"}:
        raise ValueError("ground-truth scale must be native30m or upsampled10m")
    evaluated_tiles = int(metrics["evaluated_tiles"])
    if evaluated_tiles != 305:
        raise ValueError(f"formal ground-truth Test requires 305 tiles; got {evaluated_tiles}")
    confusion_matrix = metrics["confusion_matrix"]
    confusion_pixels = sum(int(value) for row in confusion_matrix for value in row)
    supervised_pixels = int(metrics["supervised_pixels"])
    if confusion_pixels != supervised_pixels:
        raise ValueError(
            "confusion matrix total does not match supervised pixel count"
        )
    aggregation = (
        "255x255 stitched class probabilities averaged over exact 3x3 support"
        if scale == "native30m"
        else "85x85 labels replicated over exact 3x3 model-grid support; bottom/right border excluded"
    )
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "evaluation_split": "test",
        "evaluation_scale": scale,
        "interpretation": "held-out independent annual 30m reference-label evaluation",
        "evaluation_reference": {
            "root": str(Path(ground_truth_root).resolve()),
            "label_name": "y_patch_30m.npy",
            "label_shape": [85, 85],
            "valid_pixels": "label != 255",
            "cropland_mask": "not applied",
            "exclude_list": "not applied",
            "threshold": 0.5,
            "aggregation": aggregation,
        },
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
    ground_truth_root = resolve_ground_truth_root(
        workspace_root, args.ground_truth_root
    )
    if not ground_truth_root.is_dir():
        raise FileNotFoundError(f"missing ground-truth label root: {ground_truth_root}")
    with args.normalization.open("r", encoding="utf-8") as stream:
        normalization = json.load(stream)

    _seed_everything(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal E0/E1/E2-W ground-truth Test evaluation requires CUDA")
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
    output_dir = reserve_ground_truth_output_directory(args.checkpoint)
    _write_json_atomic(
        output_dir / "run_snapshot.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "evaluation_split": "test",
            "evaluation_scales": ["native30m", "upsampled10m"],
            "workspace_root": str(workspace_root.resolve()),
            "ground_truth_root": str(ground_truth_root.resolve()),
            "ground_truth_label_name": "y_patch_30m.npy",
            "valid_pixels": "label != 255",
            "cropland_mask": "not applied",
            "exclude_list": "not applied",
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
        },
    )
    test_dataset = build_test_dataset(
        args.manifest, workspace_root, normalization, expected_count=305
    )
    data_config = config["data"]
    metrics_by_scale = evaluate_ground_truth_tiles(
        model,
        test_dataset,
        ground_truth_root=ground_truth_root,
        device=device,
        window_size=int(data_config["window_size"]),
        stride=int(data_config["stride"]),
        window_batch_size=16,
        amp=bool(config["training"]["amp"]),
    )
    documents = {}
    for scale, metrics in metrics_by_scale.items():
        document = build_ground_truth_evaluation_document(
            experiment_id=experiment_id,
            scale=scale,
            checkpoint=args.checkpoint,
            checkpoint_metadata=checkpoint_metadata,
            manifest_sha256=manifest_sha256,
            normalization_sha256=normalization_sha256,
            ground_truth_root=ground_truth_root,
            metrics=metrics,
        )
        _write_json_atomic(
            output_dir / scale / "test_evaluation.json", document
        )
        documents[scale] = document
    print(json.dumps(documents, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
