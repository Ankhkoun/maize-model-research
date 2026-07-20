from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.evaluate_test import FrozenTestAsset, load_frozen_checkpoint_for_test
from src.data.manifest import sha256_file


def test_loader_rejects_checkpoint_with_different_training_configuration(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "best.pt"
    config = {
        "experiment": {"id": "E0", "seed": 42},
        "model": {"kind": "toy"},
        "data": {"window_size": 24, "stride": 24, "ignore_index": 255},
        "training": {"effective_batch_size": 16, "early_stopping_patience": 12},
    }
    torch.save(
        {
            "format_version": 1,
            "model": nn.Linear(1, 1).state_dict(),
            "training_state": {
                "epoch": 13,
                "global_step": 43251,
                "best_metric": 0.9383473661376481,
            },
            "run_config": {
                "experiment": config["experiment"],
                "model": config["model"],
                "data": config["data"],
                "training": {
                    "effective_batch_size": 32,
                    "early_stopping_patience": 12,
                },
                "manifest_sha256": "MANIFEST",
                "normalization_sha256": "NORMALIZATION",
                "physical_batch_size": 16,
                "steps_per_epoch": 3327,
                "seed": 42,
            },
        },
        checkpoint,
    )
    asset = FrozenTestAsset(
        experiment_id="E0",
        checkpoint=checkpoint,
        sha256=sha256_file(checkpoint),
        epoch=13,
        validation_maize_iou=0.9383473661376481,
    )

    with pytest.raises(ValueError, match="training configuration"):
        load_frozen_checkpoint_for_test(
            checkpoint,
            nn.Linear(1, 1),
            config,
            manifest_sha256="MANIFEST",
            normalization_sha256="NORMALIZATION",
            map_location="cpu",
            asset=asset,
        )
