from pathlib import Path

import pytest

from scripts.evaluate_test import build_test_evaluation_document, parse_args


def _metrics() -> dict:
    return {
        "overall_accuracy": 0.9,
        "maize_precision": 0.8,
        "maize_recall": 0.7,
        "maize_f1": 0.75,
        "maize_iou": 0.6,
        "miou": 0.65,
        "macro_f1": 0.7,
        "kappa": 0.5,
        "area_ratio": 0.9,
        "confusion_matrix": [[2, 1], [1, 6]],
        "loss": 0.2,
        "supervised_pixels": 10,
        "evaluated_tiles": 305,
        "per_region": {},
    }


def test_parse_args_requires_explicit_config_and_checkpoint() -> None:
    args = parse_args(
        ["--config", "config.yaml", "--checkpoint", "frozen-best.pt"]
    )

    assert args.config == Path("config.yaml")
    assert args.checkpoint == Path("frozen-best.pt")
    with pytest.raises(SystemExit):
        parse_args([])


def test_build_test_evaluation_document_records_frozen_provenance() -> None:
    document = build_test_evaluation_document(
        experiment_id="E0",
        checkpoint=Path("frozen-best.pt"),
        checkpoint_metadata={
            "epoch": 13,
            "global_step": 43251,
            "validation_maize_iou": 0.9383473661376481,
            "checkpoint_sha256": "CHECKPOINT",
        },
        manifest_sha256="MANIFEST",
        normalization_sha256="NORMALIZATION",
        metrics=_metrics(),
    )

    assert document["schema_version"] == 1
    assert document["experiment_id"] == "E0"
    assert document["evaluation_split"] == "test"
    assert document["checkpoint"]["epoch"] == 13
    assert document["checkpoint"]["sha256"] == "CHECKPOINT"
    assert document["manifest_sha256"] == "MANIFEST"
    assert document["normalization_sha256"] == "NORMALIZATION"
    assert document["metrics"]["evaluated_tiles"] == 305
    assert "pseudo-label agreement" in document["interpretation"]
    assert "not independent ground-truth accuracy" in document["interpretation"]


def test_build_test_evaluation_document_requires_305_tiles_and_pixel_conservation() -> None:
    wrong_tiles = _metrics()
    wrong_tiles["evaluated_tiles"] = 304
    with pytest.raises(ValueError, match="305"):
        build_test_evaluation_document(
            experiment_id="E0",
            checkpoint=Path("frozen-best.pt"),
            checkpoint_metadata={
                "epoch": 13,
                "global_step": 43251,
                "validation_maize_iou": 0.9383473661376481,
                "checkpoint_sha256": "CHECKPOINT",
            },
            manifest_sha256="MANIFEST",
            normalization_sha256="NORMALIZATION",
            metrics=wrong_tiles,
        )

    wrong_pixels = _metrics()
    wrong_pixels["supervised_pixels"] = 11
    with pytest.raises(ValueError, match="confusion matrix"):
        build_test_evaluation_document(
            experiment_id="E0",
            checkpoint=Path("frozen-best.pt"),
            checkpoint_metadata={
                "epoch": 13,
                "global_step": 43251,
                "validation_maize_iou": 0.9383473661376481,
                "checkpoint_sha256": "CHECKPOINT",
            },
            manifest_sha256="MANIFEST",
            normalization_sha256="NORMALIZATION",
            metrics=wrong_pixels,
        )
