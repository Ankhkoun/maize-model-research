import csv
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.evaluate_test import (
    FrozenTestAsset,
    build_test_dataset,
    load_frozen_checkpoint_for_test,
    reserve_test_output_directory,
)
from src.data.manifest import sha256_file


def _write_manifest(path: Path) -> None:
    fieldnames = (
        "sample_id",
        "region_id",
        "split",
        "cube_path",
        "label_path",
        "metadata_path",
        "time_quality_path",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "validation", "test"):
            writer.writerow(
                {
                    "sample_id": split,
                    "region_id": f"r-{split}",
                    "split": split,
                    "cube_path": f"{split}/cube.npy",
                    "label_path": f"{split}/label.npy",
                    "metadata_path": f"{split}/metadata.json",
                    "time_quality_path": f"{split}/time.json",
                }
            )


def test_build_test_dataset_only_resolves_and_constructs_test(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    workspace = tmp_path / "workspace"
    test_directory = workspace / "test"
    test_directory.mkdir(parents=True)
    for name in ("cube.npy", "label.npy", "metadata.json", "time.json"):
        (test_directory / name).touch()
    created: list[list[str]] = []

    class RecordingDataset:
        def __init__(self, records, workspace_root, normalization):
            del workspace_root, normalization
            created.append([record.split for record in records])

    dataset = build_test_dataset(
        manifest,
        workspace,
        {"mean": [0.0] * 10, "std": [1.0] * 10},
        dataset_factory=RecordingDataset,
        expected_count=1,
    )

    assert dataset is not None
    assert created == [["test"]]
    assert not (workspace / "train").exists()
    assert not (workspace / "validation").exists()


def test_build_test_dataset_requires_exact_count(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    workspace = tmp_path / "workspace"
    test_directory = workspace / "test"
    test_directory.mkdir(parents=True)
    for name in ("cube.npy", "label.npy", "metadata.json", "time.json"):
        (test_directory / name).touch()

    with pytest.raises(ValueError, match="expected 2 Test records; got 1"):
        build_test_dataset(
            manifest,
            workspace,
            {"mean": [0.0] * 10, "std": [1.0] * 10},
            dataset_factory=lambda *args, **kwargs: None,
            expected_count=2,
        )


def _toy_checkpoint(tmp_path: Path) -> tuple[Path, nn.Linear, dict, FrozenTestAsset]:
    source = nn.Linear(1, 1)
    with torch.no_grad():
        source.weight.fill_(2.0)
        source.bias.fill_(3.0)
    config = {
        "experiment": {"id": "E0", "seed": 42},
        "model": {"kind": "toy"},
        "data": {"window_size": 24, "stride": 24, "ignore_index": 255},
    }
    checkpoint = tmp_path / "best.pt"
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
        checkpoint=checkpoint,
        sha256=sha256_file(checkpoint),
        epoch=13,
        validation_maize_iou=0.9383473661376481,
    )
    return checkpoint, source, config, asset


def test_load_frozen_checkpoint_loads_model_without_training_objects(
    tmp_path: Path,
) -> None:
    checkpoint, source, config, asset = _toy_checkpoint(tmp_path)
    target = nn.Linear(1, 1)

    metadata = load_frozen_checkpoint_for_test(
        checkpoint,
        target,
        config,
        manifest_sha256="MANIFEST",
        normalization_sha256="NORMALIZATION",
        map_location="cpu",
        asset=asset,
    )

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert metadata["epoch"] == 13
    assert metadata["checkpoint_sha256"] == asset.sha256


def test_load_frozen_checkpoint_rejects_hash_and_embedded_config(
    tmp_path: Path,
) -> None:
    checkpoint, _, config, asset = _toy_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="SHA256"):
        load_frozen_checkpoint_for_test(
            checkpoint,
            nn.Linear(1, 1),
            config,
            manifest_sha256="MANIFEST",
            normalization_sha256="NORMALIZATION",
            map_location="cpu",
            asset=replace(asset, sha256="0" * 64),
        )

    bad_config = copy.deepcopy(config)
    bad_config["model"]["kind"] = "different"
    with pytest.raises(ValueError, match="model configuration"):
        load_frozen_checkpoint_for_test(
            checkpoint,
            nn.Linear(1, 1),
            bad_config,
            manifest_sha256="MANIFEST",
            normalization_sha256="NORMALIZATION",
            map_location="cpu",
            asset=asset,
        )


def test_reserve_test_output_directory_is_fixed_and_one_time(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    output = reserve_test_output_directory(checkpoint)

    assert output == checkpoint.parent / "test_evaluation"
    assert output.is_dir()
    with pytest.raises(FileExistsError, match="already exists"):
        reserve_test_output_directory(checkpoint)
