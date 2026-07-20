from pathlib import Path

import pytest

from scripts.evaluate_test_ground_truth import (
    GROUND_TRUTH_RELATIVE_ROOT,
    build_ground_truth_evaluation_document,
    parse_args,
    reserve_ground_truth_output_directory,
    resolve_ground_truth_root,
)


def _metrics(supervised_pixels: int = 4) -> dict:
    return {
        "overall_accuracy": 1.0,
        "maize_precision": 1.0,
        "maize_recall": 1.0,
        "maize_f1": 1.0,
        "maize_iou": 1.0,
        "miou": 1.0,
        "macro_f1": 1.0,
        "kappa": 1.0,
        "area_ratio": 1.0,
        "confusion_matrix": [[2, 0], [0, 2]],
        "loss": 0.1,
        "supervised_pixels": supervised_pixels,
        "evaluated_tiles": 305,
        "per_region": {},
    }


def test_cli_requires_frozen_config_and_checkpoint_and_resolves_relative_root(
    tmp_path: Path,
) -> None:
    args = parse_args(["--config", "e0.yaml", "--checkpoint", "best.pt"])

    assert args.ground_truth_root == GROUND_TRUTH_RELATIVE_ROOT
    assert resolve_ground_truth_root(tmp_path, args.ground_truth_root) == (
        tmp_path / GROUND_TRUTH_RELATIVE_ROOT
    )
    with pytest.raises(SystemExit):
        parse_args([])


def test_output_is_one_time_parent_with_two_scale_subdirectories(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    output = reserve_ground_truth_output_directory(checkpoint)

    assert output == checkpoint.parent / "test_evaluation_ground_truth"
    assert (output / "native30m").is_dir()
    assert (output / "upsampled10m").is_dir()
    with pytest.raises(FileExistsError, match="already exists"):
        reserve_ground_truth_output_directory(checkpoint)


def test_document_requires_305_tiles_and_confusion_conservation(tmp_path: Path) -> None:
    document = build_ground_truth_evaluation_document(
        experiment_id="E0",
        scale="native30m",
        checkpoint=tmp_path / "best.pt",
        checkpoint_metadata={
            "epoch": 13,
            "global_step": 43251,
            "validation_maize_iou": 0.9383473661376481,
            "checkpoint_sha256": "CHECKPOINT",
        },
        manifest_sha256="MANIFEST",
        normalization_sha256="NORMALIZATION",
        ground_truth_root=tmp_path / GROUND_TRUTH_RELATIVE_ROOT,
        metrics=_metrics(),
    )

    assert document["evaluation_reference"]["valid_pixels"] == "label != 255"
    assert document["evaluation_scale"] == "native30m"
    assert document["metrics"]["evaluated_tiles"] == 305

    with pytest.raises(ValueError, match="confusion matrix"):
        build_ground_truth_evaluation_document(
            experiment_id="E0",
            scale="upsampled10m",
            checkpoint=tmp_path / "best.pt",
            checkpoint_metadata={
                "epoch": 13,
                "global_step": 43251,
                "validation_maize_iou": 0.9383473661376481,
                "checkpoint_sha256": "CHECKPOINT",
            },
            manifest_sha256="MANIFEST",
            normalization_sha256="NORMALIZATION",
            ground_truth_root=tmp_path / GROUND_TRUTH_RELATIVE_ROOT,
            metrics=_metrics(supervised_pixels=5),
        )
