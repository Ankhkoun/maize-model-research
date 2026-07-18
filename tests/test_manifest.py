import csv
from pathlib import Path

import pytest

from src.data.manifest import build_repository_manifest, load_manifest


def _touch_sample_files(workspace: Path, sample_id: str) -> None:
    cube_dir = (
        workspace
        / "03_processed_data"
        / "cubes"
        / "xinjiang_2021"
        / "cubes"
        / "2021"
        / sample_id
    )
    label_dir = (
        workspace
        / "05_pseudo_labels"
        / "sam_refined"
        / "xinjiang_vvp_mmi_confidence_corrected"
        / "conservative_v1"
        / sample_id
    )
    cube_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for name in ("x_cube.npy", "metadata.json", "time_quality.json"):
        (cube_dir / name).touch()
    (label_dir / "pseudo_label.npy").touch()


def _write_source_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "region_id", "split", "region_sample_count"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_repository_manifest_uses_relative_paths_and_counts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    for sample_id in ("sample_a", "sample_b", "sample_c"):
        _touch_sample_files(workspace, sample_id)
    source = tmp_path / "source.csv"
    _write_source_manifest(
        source,
        [
            {"sample_id": "sample_a", "region_id": "r1", "split": "train", "region_sample_count": "1"},
            {"sample_id": "sample_b", "region_id": "r2", "split": "validation", "region_sample_count": "1"},
            {"sample_id": "sample_c", "region_id": "r3", "split": "test", "region_sample_count": "1"},
        ],
    )
    output = tmp_path / "repository.csv"

    summary = build_repository_manifest(source, output, workspace)
    records = load_manifest(output, workspace)

    assert summary.counts == {"train": 1, "validation": 1, "test": 1}
    assert len(summary.source_sha256) == 64
    assert len(records) == 3
    assert not records[0].cube_path.is_absolute()
    assert records[0].cube_path.as_posix().endswith("sample_a/x_cube.npy")
    assert records[0].resolve(workspace).cube_path.is_file()


def test_manifest_rejects_duplicate_parent_sample_ids(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _touch_sample_files(workspace, "sample_a")
    source = tmp_path / "source.csv"
    _write_source_manifest(
        source,
        [
            {"sample_id": "sample_a", "region_id": "r1", "split": "train", "region_sample_count": "1"},
            {"sample_id": "sample_a", "region_id": "r2", "split": "validation", "region_sample_count": "1"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        build_repository_manifest(source, tmp_path / "repository.csv", workspace)


def test_load_manifest_can_forbid_test_records(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _touch_sample_files(workspace, "sample_a")
    source = tmp_path / "source.csv"
    _write_source_manifest(
        source,
        [{"sample_id": "sample_a", "region_id": "r1", "split": "test", "region_sample_count": "1"}],
    )
    output = tmp_path / "repository.csv"
    build_repository_manifest(source, output, workspace)

    with pytest.raises(ValueError, match="forbidden split.*test"):
        load_manifest(output, workspace, allowed_splits={"train", "validation"})
