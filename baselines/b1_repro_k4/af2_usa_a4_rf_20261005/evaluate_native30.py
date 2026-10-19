"""Frozen USA Native30m-v1 full-grid evaluator.

The model consumes EPSG:3857 10 m 256x256 tiles. Predictions are converted
to class-1 probability, reprojected to the USDA CDL EPSG:5070 native 30 m
grid, and scored on all CDL cells with complete source-tile coverage. No
cropland mask or test pseudo-label is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from affine import Affine

import os
os.environ.setdefault("GDAL_MEM_ENABLE_OPEN", "YES")
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from .data import USAAssets, check_protocol, load_rows
from .model import AgriFMA4USA


PROTOCOL = "Native30m-v1"
EXPECTED_VALID_PIXELS = 1_959_211
EXPECTED_CDL_SHA256 = "100f0b664367453ec38e28e1e72812af66b6308f29b5c2f166acdb708940b922"

SOURCE_CRS_WKT = (
    'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",'
    'DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],'
    'PARAMETER["scale_factor",1],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","3857"]]'
)
TARGET_CRS_WKT = (
    'PROJCS["NAD83 / Conus Albers",GEOGCS["NAD83",'
    'DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",23],'
    'PARAMETER["longitude_of_center",-96],PARAMETER["standard_parallel_1",29.5],'
    'PARAMETER["standard_parallel_2",45.5],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","5070"]]'
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    total = max(tp + fp + fn + tn, 1)
    corn_iou = tp / max(tp + fp + fn, 1)
    noncorn_iou = tn / max(tn + fp + fn, 1)
    corn_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    noncorn_f1 = 2 * tn / max(2 * tn + fp + fn, 1)
    oa = (tp + tn) / total
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (total * total)
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "corn_iou": corn_iou, "noncorn_iou": noncorn_iou,
        "miou": (corn_iou + noncorn_iou) / 2.0,
        "corn_f1": corn_f1, "noncorn_f1": noncorn_f1,
        "macro_f1": (corn_f1 + noncorn_f1) / 2.0,
        "oa": oa, "kappa": (oa - pe) / max(1.0 - pe, 1e-12),
        "valid_pixels": int(total),
    }


def score(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    valid = target != 255
    tp = int(((prediction == 1) & (target == 1) & valid).sum())
    fp = int(((prediction == 1) & (target == 0) & valid).sum())
    fn = int(((prediction == 0) & (target == 1) & valid).sum())
    tn = int(((prediction == 0) & (target == 0) & valid).sum())
    return metrics(tp, fp, fn, tn)


def native_arrays(probability: np.ndarray, source_transform: tuple[float, ...], cdl: Any) -> dict[str, Any]:
    if probability.shape != (256, 256):
        raise ValueError(f"expected 256x256 probability, got {probability.shape}")
    source = Affine(*map(float, source_transform[:6]))
    bounds = transform_bounds(SOURCE_CRS_WKT, TARGET_CRS_WKT,
                              *array_bounds(256, 256, source), densify_pts=41)
    float_window = from_bounds(*bounds, cdl.transform)
    col0 = math.floor(float_window.col_off) - 1
    row0 = math.floor(float_window.row_off) - 1
    window = Window(
        col0, row0,
        math.ceil(float_window.col_off + float_window.width) - col0 + 1,
        math.ceil(float_window.row_off + float_window.height) - row0 + 1,
    )
    code = cdl.read(1, window=window)
    target_transform = cdl.window_transform(window)
    source_padded = np.pad(probability.astype(np.float32), 8)
    coverage_source = np.pad(np.ones((256, 256), dtype=np.float32), 8)
    source_shifted = source * Affine.translation(-8, -8)
    destination = np.zeros(code.shape, dtype=np.float32)
    coverage = np.zeros(code.shape, dtype=np.float32)
    reproject(source_padded, destination, src_transform=source_shifted,
              src_crs=SOURCE_CRS_WKT, dst_transform=target_transform,
              dst_crs=TARGET_CRS_WKT, resampling=Resampling.average,
              src_nodata=None, dst_nodata=0, ERROR_THRESHOLD=0.0)
    reproject(coverage_source, coverage, src_transform=source_shifted,
              src_crs=SOURCE_CRS_WKT, dst_transform=target_transform,
              dst_crs=TARGET_CRS_WKT, resampling=Resampling.average,
              src_nodata=None, dst_nodata=0, ERROR_THRESHOLD=0.0)
    valid = (code != 0) & (coverage >= 1.0 - 1e-6)
    target = np.where(valid, (code == 1).astype(np.uint8), 255).astype(np.uint8)
    prediction = (destination >= 0.5).astype(np.uint8)
    rows, cols = np.indices(code.shape)
    ids = ((row0 + rows) * cdl.width + (col0 + cols)).astype(np.int64)
    return {
        "prediction": prediction, "target": target, "ids": ids, "valid": valid,
        "source_transform": tuple(source)[:6], "target_transform": tuple(target_transform)[:6],
        "coverage": coverage,
        "window": [int(row0), int(col0), int(window.height), int(window.width)],
    }


@torch.inference_mode()
def predict_probability(model: AgriFMA4USA, assets: USAAssets,
                        row: dict[str, str], device: torch.device) -> np.ndarray:
    # Test pseudo-labels are deliberately not loaded. Neutral metadata keeps
    # the reliability/residual path usable without label leakage.
    sample = assets.load_sample(row, need_label=False)
    batch = sample["agrifm_cube"].unsqueeze(0).to(device, non_blocking=True)
    pseudo_shape = (1, 16, 1024)
    pseudo_class = torch.zeros(pseudo_shape, dtype=torch.long, device=device)
    pseudo_valid = torch.zeros(pseudo_shape, dtype=torch.bool, device=device)
    pseudo_purity = torch.zeros(pseudo_shape, dtype=torch.float32, device=device)
    pseudo_fraction = torch.zeros(pseudo_shape, dtype=torch.float32, device=device)
    output = model(
        batch, sample["a4_cube"].unsqueeze(0).to(device, non_blocking=True),
        doy=sample["doy"].unsqueeze(0).to(device, non_blocking=True),
        observed_support=sample["observed_support"].unsqueeze(0).to(device, non_blocking=True),
        pseudo_class=pseudo_class, pseudo_class_valid=pseudo_valid,
        pseudo_purity=pseudo_purity, pseudo_valid_fraction=pseudo_fraction,
        window_chunk_size=1, reliability_enabled=True,
    )
    logits = output["logits"]
    if tuple(logits.shape) != (1, 2, 256, 256) or not torch.isfinite(logits).all():
        raise ValueError(f"invalid classifier output for {row['sample_id']}: {tuple(logits.shape)}")
    return logits.float().softmax(1)[0, 1].cpu().numpy().astype(np.float32)


def _deduplicate(ids: np.ndarray, prediction: np.ndarray,
                 target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, first = np.unique(ids, return_index=True)
    first = np.sort(first)
    return prediction[first], target[first]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cdl", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    assets_dir = Path(cfg["data"]["protocol_assets"])
    assets = USAAssets(Path(cfg["data"]["root"]), assets_dir / "normalization_train1296.json")
    protocol_hashes = check_protocol(assets_dir, include_test=True)
    rows = load_rows(assets_dir / "test500.csv")
    if len(rows) != 500 or any(row.get("split") != "test" for row in rows):
        raise ValueError(f"expected exactly 500 Test rows, got {len(rows)}")
    if len({row["sample_id"] for row in rows}) != 500:
        raise ValueError("duplicate Test500 sample IDs")
    if not args.cdl.is_file():
        raise FileNotFoundError(args.cdl)
    cdl_sha256 = digest(args.cdl)
    if cdl_sha256 != EXPECTED_CDL_SHA256:
        raise ValueError(f"CDL SHA256 mismatch: {cdl_sha256}; expected {EXPECTED_CDL_SHA256}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Native30m-v1 requires CUDA")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = AgriFMA4USA(
        cfg["data"]["agrifm_source"], cfg["data"]["pretrained"],
        use_checkpointing=bool(cfg["model"].get("use_checkpointing", True)),
        use_support_features=bool(cfg["model"].get("use_support_features", True)),
        use_patch_metadata=bool(cfg["model"].get("use_patch_metadata", True)),
        warmup_backbone=False,
        feature_residual_variant=str(cfg["model"].get("feature_residual_variant", "legacy")),
        reliability_input=str(cfg["model"].get("reliability_input", "a4_cube")),
        num_experts=int(cfg["model"].get("num_experts", 4)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    checkpoint_stage = str(payload.get("stage", "A"))
    checkpoint_metrics = payload.get("metrics", {})
    stage_epoch = int(checkpoint_metrics.get("stage_epoch", 1))
    model.set_stage(checkpoint_stage)
    residual_max = float(cfg["model"].get("feature_residual_max", 0.10))
    residual_ratio_max = float(cfg["model"].get("feature_residual_ratio_max", 0.0))
    residual_ramp = max(1, int(cfg["training"].get("feature_residual_ramp_epochs", 3)))
    if bool(cfg["model"].get("enable_feature_residual", False)) and checkpoint_stage in {"J", "C", "D"}:
        feature_residual_scale = residual_max * min(1.0, stage_epoch / residual_ramp)
        ratio_schedule = cfg["training"].get("feature_residual_ratio_schedule")
        if ratio_schedule:
            schedule = [float(value) for value in ratio_schedule]
            index = min(stage_epoch - 1, len(schedule) - 1) if checkpoint_stage == "J" else len(schedule) - 1
            residual_ratio_max = schedule[index]
    else:
        feature_residual_scale = 0.0
    model.set_feature_residual_scale(feature_residual_scale)
    model.set_feature_residual_ratio_max(residual_ratio_max)
    model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output / "predictions_native30m"
    reference_dir = args.output / "references"
    prediction_dir.mkdir(exist_ok=True)
    reference_dir.mkdir(exist_ok=True)
    all_ids: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    regional: dict[str, list[list[np.ndarray]]] = defaultdict(lambda: [[], [], []])
    tile_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    with rasterio.open(args.cdl) as cdl:
        if abs(cdl.transform.a - 30.0) > 1e-6 or abs(cdl.transform.e + 30.0) > 1e-6:
            raise ValueError(f"Native30m-v1 requires 30 m CDL pixels, got {cdl.transform}")
        for index, row in enumerate(rows, 1):
            probability = predict_probability(model, assets, row, device)
            metadata = json.loads((assets.cube_dir(row["sample_id"]) / "metadata.json").read_text(encoding="utf-8"))
            native = native_arrays(probability, tuple(metadata["transform"]), cdl)
            valid = native["valid"]
            ids = native["ids"][valid]
            prediction = native["prediction"][valid]
            target = native["target"][valid]
            all_ids.append(ids); all_predictions.append(prediction); all_targets.append(target)
            regional[row["region_id"]][0].append(ids)
            regional[row["region_id"]][1].append(prediction)
            regional[row["region_id"]][2].append(target)
            np.savez_compressed(prediction_dir / f"{row['sample_id']}.npz", prediction=prediction, ids=ids)
            np.savez_compressed(reference_dir / f"{row['sample_id']}.npz", target=target, ids=ids)
            tile_records.append({"sample_id": row["sample_id"], "region_id": row["region_id"],
                                 "raw_valid_pixels": int(valid.sum()), "window": native["window"]})
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(json.dumps({"event": "native30_progress", "tiles": index, "total": len(rows),
                                  "raw_valid_pixels": int(sum(map(len, all_ids))),
                                  "seconds": time.perf_counter() - started}), flush=True)

    ids = np.concatenate(all_ids)
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    predictions, targets = _deduplicate(ids, predictions, targets)
    if len(predictions) != EXPECTED_VALID_PIXELS:
        raise ValueError(f"valid pixel count mismatch: {len(predictions)}; expected {EXPECTED_VALID_PIXELS}")
    overall = score(predictions, targets)
    per_region: dict[str, dict[str, float | int]] = {}
    for name, (region_ids, region_predictions, region_targets) in sorted(regional.items()):
        rid = np.concatenate(region_ids)
        rpred = np.concatenate(region_predictions)
        rtgt = np.concatenate(region_targets)
        rpred, rtgt = _deduplicate(rid, rpred, rtgt)
        per_region[name] = score(rpred, rtgt)

    evaluator_hash = digest(Path(__file__))
    protocol = {
        "name": PROTOCOL, "test_manifest_sha256": protocol_hashes["test500.csv"],
        "cdl_sha256": cdl_sha256, "normalization_sha256": protocol_hashes["normalization_train1296.json"],
        "input_crs": "EPSG:3857", "input_grid": "256x256 at 10 m",
        "evaluation_crs": "EPSG:5070", "evaluation_grid": "USDA CDL native 30 m",
        "mask": "cdl_code != 0 and complete source coverage >= 1 - 1e-6; no crop mask",
        "prediction": "class-1 probability from frozen AF2 full-tile model",
        "native_resampling": "area average", "threshold": ">= 0.5",
        "deduplication": "global EPSG:5070 row/column cell id",
        "expected_valid_pixels": EXPECTED_VALID_PIXELS, "valid_pixels": int(len(predictions)),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest(args.checkpoint),
        "evaluator_sha256": evaluator_hash, "test_pseudo_labels_loaded": False,
        "checkpoint_stage": checkpoint_stage, "checkpoint_stage_epoch": stage_epoch,
        "feature_residual_scale": feature_residual_scale,
    }
    result = {"status": "completed", "protocol": PROTOCOL, "test_tiles": 500,
              "overall": overall, "per_region": per_region,
              "protocol_metadata": protocol, "elapsed_seconds": time.perf_counter() - started}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output / "per_tile.jsonl").write_text("".join(json.dumps(x) + "\n" for x in tile_records), encoding="utf-8")
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                   "rasterio": rasterio.__version__, "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(device)}
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (args.output / "checkpoint_sha256.txt").write_text(protocol["checkpoint_sha256"] + "\n", encoding="ascii")
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
