import csv
from pathlib import Path

import pytest
import yaml

import scripts.train_e0 as train_e0
from scripts.train_e0 import (
    build_train_validation_datasets,
    load_formal_config,
    optimizer_steps_per_epoch,
)


ROOT = Path(__file__).resolve().parents[1]


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


def test_formal_config_freezes_approved_e0_training_parameters() -> None:
    config = load_formal_config(ROOT / "configs" / "models" / "tsvit_baseline.yaml")

    assert config["model"]["image_size"] == 24
    assert config["model"]["patch_size"] == 2
    assert config["training"] == {
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


def test_dataset_builder_never_resolves_or_constructs_test(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    workspace = tmp_path / "workspace"
    for split in ("train", "validation"):
        directory = workspace / split
        directory.mkdir(parents=True)
        for name in ("cube.npy", "label.npy", "metadata.json", "time.json"):
            (directory / name).touch()
    created: list[list[str]] = []

    class RecordingDataset:
        def __init__(self, records, workspace_root, normalization):
            del workspace_root, normalization
            created.append([record.split for record in records])

    train, validation = build_train_validation_datasets(
        manifest,
        workspace,
        {"mean": [0.0] * 10, "std": [1.0] * 10},
        dataset_factory=RecordingDataset,
        expected_counts=None,
    )

    assert train is not None and validation is not None
    assert created == [["train"], ["validation"]]
    assert not (workspace / "test").exists()


def test_optimizer_step_count_respects_parent_tile_groups(tmp_path: Path) -> None:
    import numpy as np

    labels = []
    for index, valid_windows in enumerate((17, 1)):
        label = np.full((8, 8), 255, dtype=np.uint8)
        label.reshape(-1)[:valid_windows] = 1
        path = tmp_path / f"label-{index}.npy"
        np.save(path, label)
        labels.append(path)

    assert optimizer_steps_per_epoch(labels, window_size=1, stride=1, effective_batch=16) == 3


def test_config_rejects_accidental_wpe_for_e0(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs" / "models" / "tsvit_baseline.yaml").read_text())
    config["model"]["wavelet"]["enabled"] = True
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="WPE disabled"):
        load_formal_config(path)


def test_formal_e1_config_is_accepted_with_frozen_amp_retry() -> None:
    config = load_formal_config(ROOT / "configs" / "models" / "tsvit_wpe_basic.yaml")

    assert config["experiment"]["id"] == "E1"
    assert config["model"]["wavelet"]["enabled"] is True
    assert config["training"]["amp_backoff_factor"] == 0.5
    assert config["training"]["amp_min_scale"] == 128.0
    assert config["training"]["amp_max_backoffs_per_batch"] == 6
    assert train_e0.formal_output_for(config).name == "e1_tsvit_doy_wpe_seed42"


def test_formal_e1_rejects_disabled_wpe(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "models" / "tsvit_wpe_basic.yaml").read_text(encoding="utf-8")
    )
    config["model"]["wavelet"]["enabled"] = False
    path = tmp_path / "bad-e1.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="E1 requires WPE enabled"):
        load_formal_config(path)


def test_output_dir_uses_cli_override_or_experiment_default(tmp_path: Path) -> None:
    config = load_formal_config(ROOT / "configs" / "models" / "tsvit_wpe_basic.yaml")
    requested = tmp_path / "explicit-e1-output"

    assert train_e0.resolve_output_dir(config, requested) == requested
    assert train_e0.resolve_output_dir(config, None) == train_e0.formal_output_for(config)
