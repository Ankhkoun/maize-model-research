"""USA Test500 dataset, fixed manifest checks, dual normalization, and time support."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .phenology_targets import (
    build_phase_targets,
    tile_to_windows_phase,
    tile_to_windows_scalar,
    windows_to_tile_phase,
    windows_to_tile_scalar,
)


EXPECTED_SHA256 = {
    "train_validation500.csv": "66ce0fc4dd3c95c106133f10547364400d6e7bbade0b6943991444025a9e21d4",
    "test500.csv": "1dd0984d9bdff4a580ff5db44a0a276f65c45ad853f0b85e3a8a7480baf6aaa3",
    "gtdev60.csv": "bdbc4927798889fc4fa8ee497962f14cc405d228a4ad9582274e306c0fec0e0f",
    "all_candidates500.csv": "d85fc18a21f3f7284ad13a7c70aad832dee69b6d3c5a2cdd70f198222440db97",
    "normalization_train1296.json": "e0b61bef8e7672397062c4edcaa793f75ab8bdc7f4b78cb6f62e2c5580fe514b",
}

# Official AgriFM Sentinel-2 statistics are in reflectance x10,000 units.
AGRIFM_MEAN = np.asarray([
    4179.192015478227, 4065.9106675194444, 3957.274910960156,
    5207.452475253116, 4327.12234687, 4873.16102239, 5049.1637925,
    5111.07806856, 3056.86349163, 2490.9675032,
], dtype=np.float32)
AGRIFM_STD = np.asarray([
    4041.5212325268735, 3691.003119315892, 3629.331318356375,
    2973.5178530908756, 3569.73343885, 3085.9151435, 2937.56005119,
    2806.04462314, 1808.30013156, 1694.20220774,
], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_protocol(asset_dir: Path, *, include_test: bool = False) -> dict[str, str]:
    checked = {}
    names = list(EXPECTED_SHA256) if include_test else [
        "train_validation500.csv", "gtdev60.csv", "all_candidates500.csv",
        "normalization_train1296.json",
    ]
    for name in names:
        path = asset_dir / name
        actual = sha256(path)
        expected = EXPECTED_SHA256[name]
        if actual != expected:
            raise ValueError(f"frozen USA asset hash mismatch for {name}: {actual}")
        checked[name] = actual
    return checked


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sample_id in {path}")
    return rows


def tile_grid_to_windows(grid: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """[B?,128,128] patch grid -> [B?,16,1024], row-major 4x4 windows."""
    if grid.shape[-2:] != (128, 128):
        raise ValueError(f"expected 128x128 patch grid, got {tuple(grid.shape)}")
    prefix = grid.shape[:-2]
    if len(prefix) == 0:
        return grid.reshape(4, 32, 4, 32).transpose(0, 2, 1, 3).reshape(16, 1024) if isinstance(grid, np.ndarray) else grid.reshape(4, 32, 4, 32).permute(0, 2, 1, 3).reshape(16, 1024)
    if len(prefix) != 1:
        raise ValueError("tile patch grids may have at most one batch dimension")
    if isinstance(grid, np.ndarray):
        return grid.reshape(*prefix, 4, 32, 4, 32).transpose(0, 1, 3, 2, 4).reshape(*prefix, 16, 1024)
    return grid.reshape(*prefix, 4, 32, 4, 32).permute(0, 1, 3, 2, 4).reshape(*prefix, 16, 1024)


def windows_to_tile_grid(windows: torch.Tensor) -> torch.Tensor:
    if windows.ndim != 3 or windows.shape[1:] != (16, 1024):
        raise ValueError("window reliability must be [B,16,1024]")
    b = windows.shape[0]
    return windows.reshape(b, 4, 4, 32, 32).permute(0, 1, 3, 2, 4).reshape(b, 128, 128)


def patch_targets(label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if label.shape != (256, 256) or not np.isin(label, (0, 1, 255)).all():
        raise ValueError("pseudo label must be 256x256 with values 0,1,255")
    blocks = label.reshape(128, 2, 128, 2)
    count = (blocks != 255).sum(axis=(1, 3))
    positive = (blocks == 1).sum(axis=(1, 3))
    ratio = positive / np.maximum(count, 1)
    corn = (count >= 3) & (ratio >= 0.75)
    noncorn = (count >= 3) & (ratio <= 0.25)
    valid = corn | noncorn
    purity = np.maximum(ratio, 1.0 - ratio).astype(np.float32)
    valid_fraction = (count / 4.0).astype(np.float32)
    purity[count == 0] = 0.0
    return (corn.astype(np.int64), valid, purity, valid_fraction)


def _windows_to_tile_frames(windows: torch.Tensor) -> torch.Tensor:
    """Convert [26,16,1024] window patches to [26,128,128] grids."""
    if windows.shape != (26, 16, 1024):
        raise ValueError(f"expected [26,16,1024], got {tuple(windows.shape)}")
    return windows.reshape(26, 4, 4, 32, 32).permute(0, 1, 3, 2, 4).reshape(26, 128, 128)


def _tile_frames_to_windows(tile: torch.Tensor) -> torch.Tensor:
    """Convert [26,128,128] grids to [16,1024,26] window patches."""
    if tile.shape != (26, 128, 128):
        raise ValueError(f"expected [26,128,128], got {tuple(tile.shape)}")
    return tile.reshape(26, 4, 32, 4, 32).permute(1, 3, 2, 4, 0).reshape(16, 1024, 26)


def spatial_transform(value: torch.Tensor, transform: tuple[int, bool, bool]) -> torch.Tensor:
    k, flip_h, flip_v = transform
    value = torch.rot90(value, int(k), dims=(-2, -1))
    if flip_h:
        value = value.flip(-1)
    if flip_v:
        value = value.flip(-2)
    return value.contiguous()


def _augment_sample(sample: dict[str, Any], *, generator: torch.Generator) -> dict[str, Any]:
    """Apply one shared dihedral transform to all spatially aligned assets."""
    k = int(torch.randint(0, 4, (), generator=generator))
    flip_h = bool(torch.randint(0, 2, (), generator=generator))
    flip_v = bool(torch.randint(0, 2, (), generator=generator))

    transform = (k, flip_h, flip_v)
    sample["spatial_transform"] = transform

    def spatial(value: torch.Tensor) -> torch.Tensor:
        return spatial_transform(value, transform)

    for key in ("agrifm_cube", "a4_cube", "target", "valid_pixel_mask", "source_map"):
        sample[key] = spatial(sample[key])
    support_tile = spatial(_windows_to_tile_frames(sample["observed_support"].permute(2, 0, 1)))
    sample["observed_support"] = _tile_frames_to_windows(support_tile)
    if {"phase_target", "phase_valid", "phase_confidence"} <= set(sample):
        phase_tile = spatial_transform(
            torch.from_numpy(windows_to_tile_phase(sample["phase_target"].numpy())), transform
        )
        phase_valid_tile = spatial_transform(
            torch.from_numpy(windows_to_tile_scalar(sample["phase_valid"].numpy())), transform
        )
        phase_confidence_tile = spatial_transform(
            torch.from_numpy(windows_to_tile_scalar(sample["phase_confidence"].numpy())), transform
        )
        sample["phase_target"] = torch.from_numpy(tile_to_windows_phase(phase_tile.numpy()))
        sample["phase_valid"] = torch.from_numpy(tile_to_windows_scalar(phase_valid_tile.numpy()))
        sample["phase_confidence"] = torch.from_numpy(tile_to_windows_scalar(phase_confidence_tile.numpy()))
    pseudo_class, pseudo_valid, purity, valid_fraction = patch_targets(sample["target"].numpy())
    sample["pseudo_class"] = torch.from_numpy(tile_grid_to_windows(pseudo_class))
    sample["pseudo_class_valid"] = torch.from_numpy(tile_grid_to_windows(pseudo_valid))
    sample["pseudo_purity"] = torch.from_numpy(tile_grid_to_windows(purity))
    sample["pseudo_valid_fraction"] = torch.from_numpy(tile_grid_to_windows(valid_fraction))
    return sample


def _calendar_doy(quality_path: Path) -> np.ndarray:
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if not isinstance(quality, list) or len(quality) != 26:
        raise ValueError(f"expected 26 time-quality entries: {quality_path}")
    output = []
    for index, item in enumerate(quality):
        start = dt.date(2021, 4, 1) + dt.timedelta(days=7 * index)
        end = dt.date(2021, 9, 30) if index == 25 else start + dt.timedelta(days=6)
        if item.get("slot_id") != f"t{index + 1:02d}":
            raise ValueError(f"wrong temporal slot in {quality_path}: {item}")
        match = re.search(r"_t\d{2}_(\d{8})_(\d{8})\.tif$", item.get("path", ""))
        if match:
            recorded_start = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
            recorded_end = dt.datetime.strptime(match.group(2), "%Y%m%d").date()
            if (recorded_start, recorded_end) != (start, end):
                raise ValueError(f"recorded dates disagree with USA 2021 calendar: {quality_path}")
        output.append((start.timetuple().tm_yday + end.timetuple().tm_yday) / 2.0)
    return np.asarray(output, dtype=np.float32)


class USAAssets:
    def __init__(self, workspace: Path, normalization_path: Path):
        self.workspace = workspace.resolve()
        self.cube_root = self.workspace / "03_processed_data/cubes/usa_2021/cubes_linear_interp/2021"
        self.pseudo_root = self.workspace / "05_pseudo_labels/sam_refined/usa_2021/vvp_mmi_intersection"
        self.mask_root = self.workspace / "03_processed_data/cropland_masks/usa_2021/2021"
        self.reference_root = self.workspace / "02_raw_data/reference_products/usa_reference_labels_2021_2024/2021"
        norm = json.loads(normalization_path.read_text(encoding="utf-8"))
        if norm.get("train_count") != 1296 or len(norm.get("source_hashes", {})) != 1296:
            raise ValueError("normalization_train1296 must describe exactly 1296 Train IDs")
        self.a4_mean = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
        self.a4_std = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
        if self.a4_mean.shape != (1, 10, 1, 1) or not np.isfinite(self.a4_std).all() or (self.a4_std <= 0).any():
            raise ValueError("invalid A4 normalization statistics")

    def cube_dir(self, sample_id: str) -> Path:
        path = self.cube_root / sample_id
        if not (path / "x_cube.npy").is_file():
            raise FileNotFoundError(f"cube missing for {sample_id}: {path}")
        return path

    def pseudo_path(self, sample_id: str) -> Path:
        path = self.pseudo_root / sample_id / "pseudo_label.npy"
        if not path.is_file():
            raise FileNotFoundError(f"intersection pseudo-label missing for {sample_id}: {path}")
        return path

    def crop_mask_path(self, sample_id: str) -> Path:
        path = self.mask_root / sample_id / "cropland_mask.npy"
        if not path.is_file():
            raise FileNotFoundError(f"cropland mask missing for {sample_id}: {path}")
        return path

    def reference_path(self, sample_id: str) -> Path:
        path = self.reference_root / sample_id / f"{sample_id}_cdl_corn_reference_10m.tif"
        if not path.is_file():
            raise FileNotFoundError(f"CDL reference missing for {sample_id}: {path}")
        return path

    def load_sample(self, row: dict[str, str], *, need_label: bool = True,
                    need_reference: bool = False) -> dict[str, Any]:
        sample_id = row["sample_id"]
        tile_dir = self.cube_dir(sample_id)
        raw = np.load(tile_dir / "x_cube.npy", mmap_mode="r", allow_pickle=False)
        if raw.shape != (26, 10, 256, 256) or raw.dtype != np.float32:
            raise ValueError(f"invalid cube for {sample_id}: {raw.shape} {raw.dtype}")
        raw = np.asarray(raw)
        if not np.isfinite(raw).all():
            raise ValueError(f"nonfinite original cube for {sample_id}; refusing to sanitize inputs")
        # Cube assets are reflectance in [0,1]; AgriFM published stats expect [0,10000].
        agrifm = (raw * 10000.0 - AGRIFM_MEAN[None, :, None, None]) / AGRIFM_STD[None, :, None, None]
        a4 = (raw - self.a4_mean) / self.a4_std
        source_map = np.load(tile_dir / "source_map.npy", mmap_mode="r", allow_pickle=False)
        # USA source_map uses the frozen 1..4 quality codes: 1 is a direct
        # observation, 2/3 are valid derived/interpolated sources, and 4 is
        # unresolved.  WPE observed support below intentionally counts only
        # code 1; accepting the other codes here must not turn them into
        # observed support.
        if source_map.shape != (26, 256, 256) or source_map.dtype != np.uint8 or not np.isin(source_map, (1, 2, 3, 4)).all():
            raise ValueError(f"invalid source_map for {sample_id}")
        meta = json.loads((tile_dir / "metadata.json").read_text(encoding="utf-8"))
        if meta.get("sample_id") != sample_id or meta.get("crs") != "EPSG:3857":
            raise ValueError(f"cube metadata identity/CRS mismatch for {sample_id}")
        doy = _calendar_doy(tile_dir / "time_quality.json")
        patch_support = (source_map == 1).reshape(26, 128, 2, 128, 2).mean(axis=(2, 4))
        support = (patch_support.reshape(26, 4, 32, 4, 32)
                   .transpose(1, 3, 2, 4, 0).reshape(16, 1024, 26)
                   .astype(np.float32, copy=False))
        result: dict[str, Any] = {
            "sample_id": sample_id,
            "region_id": row["region_id"],
            "agrifm_cube": torch.from_numpy(np.ascontiguousarray(agrifm)),
            "a4_cube": torch.from_numpy(np.ascontiguousarray(a4)),
            "doy": torch.from_numpy(doy),
            "observed_support": torch.from_numpy(np.ascontiguousarray(support)),
            "source_map": torch.from_numpy(np.asarray(source_map).copy()),
            "cube_metadata": meta,
        }
        if need_label:
            label = np.load(self.pseudo_path(sample_id), allow_pickle=False)
            if label.shape != (256, 256) or not np.isin(label, (0, 1, 255)).all():
                raise ValueError(f"invalid pseudo-label for {sample_id}")
            pseudo_class, pseudo_valid, pseudo_purity, pseudo_valid_fraction = patch_targets(label)
            phase_target, phase_valid, phase_confidence, phase_report = build_phase_targets(
                raw, support, tile_grid_to_windows(pseudo_class), tile_grid_to_windows(pseudo_valid)
            )
            result["target"] = torch.from_numpy(label.astype(np.int64, copy=True))
            result["valid_pixel_mask"] = torch.from_numpy(label != 255)
            result["pseudo_class"] = torch.from_numpy(tile_grid_to_windows(pseudo_class))
            result["pseudo_class_valid"] = torch.from_numpy(tile_grid_to_windows(pseudo_valid))
            result["pseudo_purity"] = torch.from_numpy(tile_grid_to_windows(pseudo_purity))
            result["pseudo_valid_fraction"] = torch.from_numpy(tile_grid_to_windows(pseudo_valid_fraction))
            result["phase_target"] = torch.from_numpy(phase_target)
            result["phase_valid"] = torch.from_numpy(phase_valid)
            result["phase_confidence"] = torch.from_numpy(phase_confidence)
            result["phase_report"] = phase_report
        if need_reference:
            result["cropland_mask"] = torch.from_numpy(np.load(self.crop_mask_path(sample_id), allow_pickle=False))
            result["cdl_reference_path"] = str(self.reference_path(sample_id))
        return result


class USATrainDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], assets: USAAssets,
                 split: str, *, need_reference: bool = False, augment: bool = False,
                 seed: int = 42):
        self.rows = [row for row in rows if row["split"] == split]
        self.assets, self.split, self.need_reference = assets, split, need_reference
        self.augment = bool(augment and split == "train")
        # Windows DataLoader workers use spawn; a torch.Generator cannot be
        # pickled as part of the Dataset.  Create one lazily inside each
        # worker so augmentation remains deterministic without blocking
        # multiprocessing.
        self.seed = int(seed)
        self._generator: torch.Generator | None = None
        if split not in {"train", "validation"}:
            raise ValueError("training dataset accepts only Train or Validation")
        if not self.rows:
            raise ValueError(f"no {split} rows")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.assets.load_sample(self.rows[index], need_label=True,
                                         need_reference=self.need_reference)
        if self.augment:
            if self._generator is None:
                worker = torch.utils.data.get_worker_info()
                worker_id = 0 if worker is None else int(worker.id)
                self._generator = torch.Generator().manual_seed(self.seed + 1009 * worker_id)
            sample = _augment_sample(sample, generator=self._generator)
        return sample


def collate_tiles(samples: list[dict[str, Any]]) -> dict[str, Any]:
    keys = samples[0].keys()
    batch = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values)
        elif key in {"cube_metadata", "cdl_reference_path"}:
            batch[key] = values
        else:
            batch[key] = values
    return batch


__all__ = [
    "AGRIFM_MEAN", "AGRIFM_STD", "EXPECTED_SHA256", "USAAssets",
    "USATrainDataset", "check_protocol", "collate_tiles", "load_rows",
    "patch_targets", "sha256", "tile_grid_to_windows", "windows_to_tile_grid",
]
