"""Strict full-tile loader for Xinjiang 2021 cubes and pseudo-labels."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import TileRecord


EXPECTED_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")


class XinjiangTileDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[TileRecord],
        workspace_root: Path,
        *,
        expected_frames: int = 26,
        expected_channels: int = 10,
        expected_height: int = 256,
        expected_width: int = 256,
        normalization: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.records = list(records)
        self.workspace_root = Path(workspace_root)
        self.expected_shape = (
            int(expected_frames),
            int(expected_channels),
            int(expected_height),
            int(expected_width),
        )
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.records)

    def _load_time(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        with path.open("r", encoding="utf-8") as stream:
            entries = json.load(stream)
        if not isinstance(entries, list) or len(entries) != self.expected_shape[0]:
            raise ValueError(
                f"time_quality must contain {self.expected_shape[0]} ordered slots"
            )
        doy: list[float] = []
        valid: list[bool] = []
        previous_start: date | None = None
        for index, entry in enumerate(entries, start=1):
            if entry.get("slot_id") != f"t{index:02d}":
                raise ValueError("time_quality slot_id order is invalid")
            start = date.fromisoformat(entry["start_date"])
            end = date.fromisoformat(entry["end_date"])
            if end < start or (previous_start is not None and start <= previous_start):
                raise ValueError("time_quality dates are not strictly ordered")
            previous_start = start
            midpoint = start + timedelta(days=(end - start).days // 2)
            doy.append(float(midpoint.timetuple().tm_yday))
            valid.append(int(entry.get("resolved_pixel_count", 0)) > 0)
        return np.asarray(doy, dtype=np.float32), np.asarray(valid, dtype=np.bool_)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        resolved = record.resolve(self.workspace_root)
        with resolved.metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        images = np.load(resolved.cube_path, allow_pickle=False)
        if images.shape != self.expected_shape:
            raise ValueError(
                f"cube shape for {record.sample_id} is {images.shape}, expected {self.expected_shape}"
            )
        if images.dtype != np.float32:
            raise ValueError(f"cube dtype for {record.sample_id} must be float32")
        if tuple(metadata.get("shape", ())) != self.expected_shape:
            raise ValueError(f"metadata shape disagrees for {record.sample_id}")
        if tuple(metadata.get("axis_order", ())) != ("time", "channel", "height", "width"):
            raise ValueError(f"metadata axis_order disagrees for {record.sample_id}")
        if tuple(metadata.get("band_names", ())) != EXPECTED_BANDS:
            raise ValueError(f"metadata band_names disagree for {record.sample_id}")

        doy, valid_mask = self._load_time(resolved.time_quality_path)
        if valid_mask.any() and not np.isfinite(images[valid_mask]).all():
            raise ValueError(f"cube has non-finite values in valid slots for {record.sample_id}")

        label = np.load(resolved.label_path, allow_pickle=False)
        expected_label_shape = self.expected_shape[2:]
        if label.shape != expected_label_shape:
            raise ValueError(
                f"label shape for {record.sample_id} is {label.shape}, expected {expected_label_shape}"
            )
        values = set(np.unique(label).tolist())
        if not values.issubset({0, 1, 255}):
            raise ValueError(
                f"label values for {record.sample_id} must be a subset of 0,1,255; got {sorted(values)}"
            )

        if self.normalization is not None:
            mean = np.asarray(self.normalization["mean"], dtype=np.float32)
            std = np.asarray(self.normalization["std"], dtype=np.float32)
            if mean.shape != (self.expected_shape[1],) or std.shape != mean.shape:
                raise ValueError("normalization mean/std must match the channel count")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError("normalization values must be finite with positive std")
            images = (images - mean[None, :, None, None]) / std[None, :, None, None]

        return {
            "images": torch.from_numpy(np.asarray(images, dtype=np.float32)),
            "doy": torch.from_numpy(doy),
            "valid_mask": torch.from_numpy(valid_mask),
            "label": torch.from_numpy(label.astype(np.int64, copy=False)),
            "sample_id": record.sample_id,
            "region_id": record.region_id,
            "split": record.split,
        }
