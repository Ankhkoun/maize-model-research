import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.manifest import TileRecord
from src.data.xinjiang_tiles import XinjiangTileDataset


BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def _make_record(tmp_path: Path, label_values: np.ndarray | None = None) -> tuple[TileRecord, Path]:
    workspace = tmp_path / "workspace"
    cube_dir = workspace / "cube" / "sample_a"
    label_dir = workspace / "label" / "sample_a"
    cube_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    cube = np.arange(26 * 10 * 8 * 8, dtype=np.float32).reshape(26, 10, 8, 8)
    np.save(cube_dir / "x_cube.npy", cube)
    metadata = {
        "sample_id": "sample_a",
        "shape": [26, 10, 8, 8],
        "axis_order": ["time", "channel", "height", "width"],
        "band_names": BANDS,
        "dtype": "float32",
    }
    (cube_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    slots = []
    start = date(2021, 4, 1)
    for index in range(26):
        slot_start = start + timedelta(days=7 * index)
        slots.append(
            {
                "slot_id": f"t{index + 1:02d}",
                "start_date": slot_start.isoformat(),
                "end_date": (slot_start + timedelta(days=6)).isoformat(),
                "resolved_pixel_count": 0 if index == 2 else 64,
                "unresolved_ratio": 1.0 if index == 2 else 0.0,
            }
        )
    (cube_dir / "time_quality.json").write_text(json.dumps(slots), encoding="utf-8")
    if label_values is None:
        label_values = np.zeros((8, 8), dtype=np.uint8)
        label_values[0, 0] = 1
        label_values[0, 1] = 255
    np.save(label_dir / "pseudo_label.npy", label_values)
    record = TileRecord(
        sample_id="sample_a",
        region_id="region_a",
        split="train",
        cube_path=Path("cube/sample_a/x_cube.npy"),
        label_path=Path("label/sample_a/pseudo_label.npy"),
        metadata_path=Path("cube/sample_a/metadata.json"),
        time_quality_path=Path("cube/sample_a/time_quality.json"),
    )
    return record, workspace


def test_tile_loader_validates_contract_and_uses_midpoint_doy(tmp_path: Path) -> None:
    record, workspace = _make_record(tmp_path)
    dataset = XinjiangTileDataset(
        [record], workspace, expected_height=8, expected_width=8
    )

    sample = dataset[0]

    assert sample["images"].shape == (26, 10, 8, 8)
    assert sample["images"].dtype == torch.float32
    assert sample["label"].shape == (8, 8)
    assert sample["label"].dtype == torch.int64
    assert sample["doy"].tolist() == [94.0 + 7.0 * index for index in range(26)]
    assert sample["valid_mask"].dtype == torch.bool
    assert sample["valid_mask"].sum().item() == 25
    assert not sample["valid_mask"][2]
    assert sample["sample_id"] == "sample_a"
    assert sample["region_id"] == "region_a"


def test_tile_loader_rejects_unknown_label_value(tmp_path: Path) -> None:
    label = np.zeros((8, 8), dtype=np.uint8)
    label[3, 4] = 2
    record, workspace = _make_record(tmp_path, label)
    dataset = XinjiangTileDataset(
        [record], workspace, expected_height=8, expected_width=8
    )

    with pytest.raises(ValueError, match="label values"):
        dataset[0]
