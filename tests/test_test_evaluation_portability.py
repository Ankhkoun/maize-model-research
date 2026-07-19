from pathlib import Path

import torch
from torch import nn

from scripts.evaluate_test import (
    FROZEN_TEST_ASSETS,
    FrozenTestAsset,
    load_frozen_checkpoint_for_test,
)
from src.data.manifest import sha256_file


def test_formal_frozen_checkpoint_paths_are_workspace_relative() -> None:
    for asset in FROZEN_TEST_ASSETS.values():
        assert not asset.checkpoint.is_absolute()
        assert asset.checkpoint.parts[:2] == ("06_models", "retrain_outputs")


def test_loader_resolves_relative_frozen_checkpoint_from_workspace(
    tmp_path: Path,
) -> None:
    relative = Path("06_models/retrain_outputs/run/best.pt")
    checkpoint = tmp_path / relative
    checkpoint.parent.mkdir(parents=True)
    source = nn.Linear(1, 1)
    config = {
        "experiment": {"id": "E0", "seed": 42},
        "model": {"kind": "toy"},
        "data": {"window_size": 24, "stride": 24, "ignore_index": 255},
    }
    torch.save(
        {
            "format_version": 1,
            "model": source.state_dict(),
            "training_state": {
                "epoch": 13,
                "global_step": 43251,
                "best_metric": 0.9383473661376481,
                "bad_epochs": 0,
            },
            "run_config": {
                "experiment": config["experiment"],
                "model": config["model"],
                "data": config["data"],
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
        checkpoint=relative,
        sha256=sha256_file(checkpoint),
        epoch=13,
        validation_maize_iou=0.9383473661376481,
    )

    metadata = load_frozen_checkpoint_for_test(
        checkpoint,
        nn.Linear(1, 1),
        config,
        manifest_sha256="MANIFEST",
        normalization_sha256="NORMALIZATION",
        map_location="cpu",
        asset=asset,
        workspace_root=tmp_path,
    )

    assert metadata["checkpoint_sha256"] == asset.sha256
