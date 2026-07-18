"""Repository-local manifest construction and validation."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VALID_SPLITS = frozenset({"train", "validation", "test"})


def normalize_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "val":
        normalized = "validation"
    if normalized not in VALID_SPLITS:
        raise ValueError(f"invalid split: {value}")
    return normalized
FIELDNAMES = (
    "sample_id",
    "region_id",
    "split",
    "cube_path",
    "label_path",
    "metadata_path",
    "time_quality_path",
)


@dataclass(frozen=True)
class TileRecord:
    sample_id: str
    region_id: str
    split: str
    cube_path: Path
    label_path: Path
    metadata_path: Path
    time_quality_path: Path

    def resolve(self, workspace_root: Path) -> "TileRecord":
        root = Path(workspace_root).resolve()

        def resolve_path(path: Path) -> Path:
            if path.is_absolute():
                resolved = path.resolve()
            else:
                resolved = (root / path).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"manifest path escapes workspace_root: {path}")
            return resolved

        return TileRecord(
            sample_id=self.sample_id,
            region_id=self.region_id,
            split=self.split,
            cube_path=resolve_path(self.cube_path),
            label_path=resolve_path(self.label_path),
            metadata_path=resolve_path(self.metadata_path),
            time_quality_path=resolve_path(self.time_quality_path),
        )


@dataclass(frozen=True)
class ManifestSummary:
    counts: dict[str, int]
    source_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _relative_paths(sample_id: str) -> dict[str, Path]:
    cube_dir = (
        Path("03_processed_data")
        / "cubes"
        / "xinjiang_2021"
        / "cubes"
        / "2021"
        / sample_id
    )
    label_dir = (
        Path("05_pseudo_labels")
        / "sam_refined"
        / "xinjiang_vvp_mmi_confidence_corrected"
        / "conservative_v1"
        / sample_id
    )
    return {
        "cube_path": cube_dir / "x_cube.npy",
        "label_path": label_dir / "pseudo_label.npy",
        "metadata_path": cube_dir / "metadata.json",
        "time_quality_path": cube_dir / "time_quality.json",
    }


def _validate_unique(records: Iterable[TileRecord]) -> None:
    sample_ids = [record.sample_id for record in records]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate sample_id values across splits: {duplicates[:5]}")


def build_repository_manifest(
    source_manifest: Path,
    output_manifest: Path,
    workspace_root: Path,
) -> ManifestSummary:
    source_manifest = Path(source_manifest)
    workspace_root = Path(workspace_root)
    with source_manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_id", "region_id", "split"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"source manifest missing columns: {sorted(missing)}")
        rows = list(reader)

    records: list[TileRecord] = []
    for row in rows:
        sample_id = row["sample_id"].strip()
        region_id = row["region_id"].strip()
        split = normalize_split(row["split"])
        if not sample_id or not region_id:
            raise ValueError("sample_id and region_id must be non-empty")
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split for {sample_id}: {split}")
        paths = _relative_paths(sample_id)
        record = TileRecord(sample_id, region_id, split, **paths)
        resolved = record.resolve(workspace_root)
        for field in ("cube_path", "label_path", "metadata_path", "time_quality_path"):
            path = getattr(resolved, field)
            if not path.is_file():
                raise FileNotFoundError(f"missing {field} for {sample_id}: {path}")
        records.append(record)

    _validate_unique(records)
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "region_id": record.region_id,
                    "split": record.split,
                    "cube_path": record.cube_path.as_posix(),
                    "label_path": record.label_path.as_posix(),
                    "metadata_path": record.metadata_path.as_posix(),
                    "time_quality_path": record.time_quality_path.as_posix(),
                }
            )

    counts = Counter(record.split for record in records)
    return ManifestSummary(
        counts={split: counts.get(split, 0) for split in ("train", "validation", "test")},
        source_sha256=sha256_file(source_manifest),
    )


def load_manifest(
    manifest_path: Path,
    workspace_root: Path,
    *,
    allowed_splits: set[str] | frozenset[str] | None = None,
    selected_splits: set[str] | frozenset[str] | None = None,
) -> list[TileRecord]:
    with Path(manifest_path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDNAMES).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"repository manifest missing columns: {sorted(missing)}")
        records = [
            TileRecord(
                sample_id=row["sample_id"].strip(),
                region_id=row["region_id"].strip(),
                split=normalize_split(row["split"]),
                cube_path=Path(row["cube_path"]),
                label_path=Path(row["label_path"]),
                metadata_path=Path(row["metadata_path"]),
                time_quality_path=Path(row["time_quality_path"]),
            )
            for row in reader
        ]

    _validate_unique(records)
    if selected_splits is not None:
        invalid = set(selected_splits).difference(VALID_SPLITS)
        if invalid:
            raise ValueError(f"invalid selected splits: {sorted(invalid)}")
        records = [record for record in records if record.split in selected_splits]
    for record in records:
        if record.split not in VALID_SPLITS:
            raise ValueError(f"invalid split for {record.sample_id}: {record.split}")
        if allowed_splits is not None and record.split not in allowed_splits:
            raise ValueError(
                f"forbidden split {record.split} requested for {record.sample_id}"
            )
        resolved = record.resolve(Path(workspace_root))
        for field in ("cube_path", "label_path", "metadata_path", "time_quality_path"):
            path = getattr(resolved, field)
            if not path.is_file():
                raise FileNotFoundError(f"missing {field} for {record.sample_id}: {path}")
    return records
