"""Independent AF2 USA trainer with explicit A/B/C/D stages.

The Test500 manifest is never opened here. GTdev60 is used only for checkpoint
selection through the fixed derived 85x85, 3x3 aggregation protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import USAAssets, USATrainDataset, check_protocol, collate_tiles, load_rows, spatial_transform
from .model import AgriFMA4USA
from .objective import stage_loss
from .targets import build_targets


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("AF2 USA training requires CUDA; use preflight.py for CPU checks")
    return torch.device("cuda")


def _loader(rows: list[dict[str, str]], assets: USAAssets, split: str,
            cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    selected = rows
    limit = cfg["training"].get("max_train_samples")
    if split == "train" and limit is not None:
        selected = [r for r in rows if r["split"] == split][:int(limit)]
    dataset = USATrainDataset(
        selected, assets, split,
        augment=bool(cfg["training"].get("spatial_augmentation", False) and shuffle),
        seed=int(cfg["training"].get("seed", 42)),
    )
    workers = int(cfg["training"].get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(cfg["training"].get("micro_batch", 1)),
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": collate_tiles,
        "persistent_workers": bool(workers and cfg["training"].get("persistent_workers", True)),
    }
    if workers:
        loader_kwargs["prefetch_factor"] = int(cfg["training"].get("prefetch_factor", 2))
    return DataLoader(dataset, **loader_kwargs)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: (value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value)
            for key, value in batch.items()}


@torch.no_grad()
def _refresh_target_bank(model: AgriFMA4USA, loader: DataLoader, device: torch.device,
                         previous: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
                         cfg: dict[str, Any]) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    model.eval()
    bank: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    reports: list[dict[str, Any]] = []
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, 1):
        moved = _move(batch, device)
        output = model(moved["agrifm_cube"], moved["a4_cube"],
                       doy=moved["doy"], observed_support=moved["observed_support"],
                       pseudo_class=moved["pseudo_class"],
                       pseudo_class_valid=moved["pseudo_class_valid"],
                       reliability_enabled=False)
        for index, sid in enumerate(batch["sample_id"]):
            target, valid, report = build_targets(
                output["logits"][index:index + 1], moved["target"][index:index + 1],
                previous[sid][0].unsqueeze(0).to(device) if previous and sid in previous else None,
                previous[sid][1].unsqueeze(0).to(device) if previous and sid in previous else None,
                ema=float(cfg.get("ema", 0.95)),
                target_mode=str(cfg.get("target_mode", "gmm")),
                gmm_steps=int(cfg.get("gmm_steps", 40)),
                gmm_min_samples=int(cfg.get("gmm_min_samples", 32)),
                gmm_min_separation=float(cfg.get("gmm_min_separation", 0.08)),
                gmm_min_weight=float(cfg.get("gmm_min_weight", 0.03)),
            )
            bank[sid] = (target[0].cpu(), valid[0].cpu())
            reports.append(report)
        if batch_index == 1 or batch_index % 50 == 0 or batch_index == total_batches:
            print(json.dumps({"event": "target_refresh_progress",
                              "batch": batch_index, "total_batches": total_batches,
                              "samples": len(bank)}), flush=True)
    eligible = float(np.mean([r["eligible_fraction"] for r in reports])) if reports else 0.0
    return bank, {"samples": len(bank), "eligible_fraction": eligible}


def _target_batch(batch: dict[str, Any], bank: dict[str, tuple[torch.Tensor, torch.Tensor]],
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    missing = [sid for sid in batch["sample_id"] if sid not in bank]
    if missing:
        raise KeyError(f"target bank missing samples: {missing[:3]}")
    transforms = batch.get("spatial_transform", [(0, False, False)] * len(batch["sample_id"]))
    target = torch.stack([spatial_transform(bank[sid][0], transform)
                          for sid, transform in zip(batch["sample_id"], transforms)]).to(device)
    valid = torch.stack([spatial_transform(bank[sid][1], transform)
                         for sid, transform in zip(batch["sample_id"], transforms)]).to(device)
    return target, valid


def _optimizer(model: AgriFMA4USA, stage: str, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    trainable = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = float(cfg["training"]["decoder_lr"])
        if name.startswith("reliability.") or name.startswith("multiscale_reliability."):
            lr = float(cfg["training"]["reliability_lr"])
        elif name.startswith("classifier.backbone") or name.startswith("classifier.patch_embed"):
            lr = float(cfg["training"]["backbone_lr"])
        trainable.append({"params": [parameter], "lr": lr})
    if not trainable:
        raise RuntimeError(f"stage {stage} has no trainable parameters")
    return torch.optim.AdamW(trainable, weight_decay=float(cfg["training"]["weight_decay"]))


def _load_init_checkpoint(model: AgriFMA4USA, payload: dict[str, Any]) -> dict[str, Any]:
    """Load A5 classifier weights while isolating intentional reliability migrations."""
    incoming = payload.get("model")
    if not isinstance(incoming, dict):
        raise ValueError("init checkpoint does not contain a model state dictionary")
    current = model.state_dict()
    reset_prefixes = set()
    for key, value in incoming.items():
        if key.startswith("reliability.experts.") and key in current and current[key].shape != value.shape:
            reset_prefixes.add(".".join(key.split(".")[:3]) + ".")
    old_model_cfg = payload.get("config", {}).get("model", {})
    if model.reliability.use_patch_metadata and not old_model_cfg.get("use_patch_metadata", False):
        reset_prefixes.add("reliability.patch_metadata.")
    if model.reliability.use_support_features and not old_model_cfg.get("use_support_features", False):
        reset_prefixes.add("reliability.support_embedding.")
    compatible = {}
    skipped_shape = []
    unknown = []
    for key, value in incoming.items():
        if key not in current:
            unknown.append(key)
        elif current[key].shape != value.shape:
            if key.startswith("reliability."):
                skipped_shape.append({"key": key, "checkpoint": list(value.shape),
                                      "model": list(current[key].shape)})
            else:
                raise RuntimeError(
                    f"classifier checkpoint shape mismatch for {key}: "
                    f"checkpoint={tuple(value.shape)} model={tuple(current[key].shape)}"
                )
        else:
            if not any(key.startswith(prefix) for prefix in reset_prefixes):
                compatible[key] = value
    load_result = model.load_state_dict(compatible, strict=False)
    allowed_new = ("reliability.", "stage_film.", "expert_feature_residual.")
    unexpected_missing = [key for key in load_result.missing_keys
                          if not key.startswith(allowed_new)]
    if unexpected_missing or unknown:
        raise RuntimeError(
            f"incompatible init checkpoint: missing={unexpected_missing}, unknown={unknown[:8]}"
        )
    return {
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
        "skipped_reliability_shape": skipped_shape,
        "unknown_input_keys": unknown,
        "reinitialized_modules": sorted(reset_prefixes),
    }


@torch.no_grad()
def _gtdev_target_path(assets: USAAssets, sample_id: str) -> Path:
    """Resolve the fixed GTdev reference without assuming one historical layout."""
    candidates = (
        assets.workspace / "03_processed_data/labels_30m/usa_typical_2021/2021"
        / sample_id / "y_patch_30m.npy",
        assets.workspace / "03_processed_data/labels_10m/usa_2021/2021"
        / sample_id / "y_patch.npy",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"GTdev reference missing for {sample_id}; checked: "
        + "; ".join(str(path) for path in candidates)
    )


def _load_gtdev_target(assets: USAAssets, sample_id: str) -> np.ndarray:
    """Load the fixed 85x85 derived target from 30m or 10m reference assets."""
    path = _gtdev_target_path(assets, sample_id)
    raw = np.load(path, allow_pickle=False)
    if raw.shape == (85, 85):
        return raw
    if raw.shape != (256, 256):
        raise ValueError(f"unsupported GTdev target shape: {path} {raw.shape}")
    block = raw[:255, :255].reshape(85, 3, 85, 3)
    valid = block != 255
    count = valid.sum(axis=(1, 3))
    maize = ((block == 1) & valid).sum(axis=(1, 3))
    target = np.where(count > 0, (maize / np.maximum(count, 1)) >= 0.5, 255).astype(np.uint8)
    target[count == 0] = 255
    return target


def _validate_gtdev_assets(assets: USAAssets, rows: list[dict[str, str]]) -> None:
    """Fail before training if the fixed 60-tile evaluation assets are incomplete."""
    missing = []
    for row in rows:
        try:
            target = _load_gtdev_target(assets, row["sample_id"])
        except FileNotFoundError:
            missing.append(row["sample_id"])
            continue
        if target.shape != (85, 85):
            raise ValueError(f"GTdev target shape mismatch: {row['sample_id']} {target.shape}")
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(rows)} GTdev targets are missing; "
            f"first entries: {missing[:5]}"
        )


def _metric_scores(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    total = max(tp + fp + fn + tn, 1)
    corn_iou = tp / max(tp + fp + fn, 1)
    noncorn_iou = tn / max(tn + fp + fn, 1)
    corn_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    noncorn_f1 = 2 * tn / max(2 * tn + fp + fn, 1)
    oa = (tp + tn) / total
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (total * total)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "corn_iou": corn_iou, "noncorn_iou": noncorn_iou,
        "miou": (corn_iou + noncorn_iou) / 2.0,
        "corn_recall": tp / max(tp + fn, 1),
        "corn_f1": corn_f1, "noncorn_f1": noncorn_f1,
        "macro_f1": (corn_f1 + noncorn_f1) / 2.0,
        "oa": oa, "kappa": (oa - pe) / max(1.0 - pe, 1e-12),
        "valid_pixels": total,
    }


def _aggregate_probability(probability: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Aggregate native 256px probabilities to a fixed reference grid."""
    if probability.shape == target_shape:
        return probability
    if target_shape == (85, 85):
        return probability[:255, :255].reshape(85, 3, 85, 3).mean(axis=(1, 3))
    height, width = target_shape
    if probability.shape[0] % height == 0 and probability.shape[1] % width == 0:
        row_factor = probability.shape[0] // height
        col_factor = probability.shape[1] // width
        return probability.reshape(height, row_factor, width, col_factor).mean(axis=(1, 3))
    raise ValueError(f"cannot aggregate probability {probability.shape} to {target_shape}")


def _read_binary_reference(path: Path) -> np.ndarray:
    """Read a binary CDL reference without importing rasterio at module import."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required for Validation330 threshold calibration") from exc
    with rasterio.open(path) as source:
        raw = source.read(1)
        nodata = source.nodata
    raw = np.asarray(raw)
    valid = np.isfinite(raw)
    if nodata is not None:
        valid &= raw != nodata
    valid &= raw != 255
    unique = set(np.unique(raw[valid]).tolist()) if bool(valid.any()) else set()
    if not unique.issubset({0, 1}):
        raise ValueError(f"unsupported binary CDL values in {path}: {sorted(unique)[:10]}")
    target = np.full(raw.shape, 255, dtype=np.uint8)
    target[valid] = raw[valid].astype(np.uint8)
    return target


@torch.no_grad()
def _predict_records(model: AgriFMA4USA, assets: USAAssets,
                     rows: list[dict[str, str]], device: torch.device,
                     target_loader, *, reliability_enabled: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect probabilities once so threshold scans never repeat inference."""
    model.eval()
    records = []
    for row in rows:
        sample = assets.load_sample(row, need_label=True)
        output = model(
            sample["agrifm_cube"].unsqueeze(0).to(device),
            sample["a4_cube"].unsqueeze(0).to(device),
            doy=sample["doy"].unsqueeze(0).to(device),
            observed_support=sample["observed_support"].unsqueeze(0).to(device),
            pseudo_class=sample["pseudo_class"].unsqueeze(0).to(device),
            pseudo_class_valid=sample["pseudo_class_valid"].unsqueeze(0).to(device),
            pseudo_purity=sample.get("pseudo_purity").unsqueeze(0).to(device),
            pseudo_valid_fraction=sample.get("pseudo_valid_fraction").unsqueeze(0).to(device),
            reliability_enabled=reliability_enabled,
        )
        probability = output["logits"].float().softmax(1)[0, 1].cpu().numpy()
        target = target_loader(assets, row["sample_id"])
        probability = _aggregate_probability(probability, tuple(target.shape))
        records.append((probability, target))
    return records


def _metrics_from_records(records: list[tuple[np.ndarray, np.ndarray]],
                          threshold: float) -> dict[str, float | int]:
    tp = fp = fn = tn = 0
    for probability, target in records:
        prediction = probability >= threshold
        valid = target != 255
        tp += int((prediction & (target == 1) & valid).sum())
        fp += int((prediction & (target == 0) & valid).sum())
        fn += int((~prediction & (target == 1) & valid).sum())
        tn += int((~prediction & (target == 0) & valid).sum())
    return _metric_scores(tp, fp, fn, tn) | {
        "threshold": float(threshold), "tiles": len(records),
    }


@torch.no_grad()
def _gtdev_metrics(model: AgriFMA4USA, assets: USAAssets,
                   rows: list[dict[str, str]], device: torch.device,
                   threshold: float = 0.5, *, reliability_enabled: bool = False) -> dict[str, float]:
    records = _predict_records(model, assets, rows, device, _load_gtdev_target,
                               reliability_enabled=reliability_enabled)
    return _metrics_from_records(records, threshold)


def _validation_target_path(assets: USAAssets, sample_id: str) -> Path:
    return assets.reference_path(sample_id)


def _validation_metrics(model: AgriFMA4USA, assets: USAAssets,
                        rows: list[dict[str, str]], device: torch.device,
                        *, reliability_enabled: bool = False) -> dict[str, Any]:
    """Scan Validation330 thresholds; absence of references is an explicit fallback."""
    try:
        records = _predict_records(
            model, assets, rows, device,
            lambda current_assets, sample_id: _read_binary_reference(
                _validation_target_path(current_assets, sample_id)
            ),
            reliability_enabled=reliability_enabled,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return {
            "available": False,
            "reason": str(exc),
            "thresholds": [],
            "fixed_0.50": None,
            "calibrated": None,
        }
    thresholds = [round(value, 2) for value in np.arange(0.30, 0.701, 0.01)]
    scanned = [_metrics_from_records(records, threshold) for threshold in thresholds]
    calibrated = max(scanned, key=lambda item: (float(item["miou"]), -abs(float(item["threshold"]) - 0.5)))
    fixed = _metrics_from_records(records, 0.5)
    return {
        "available": True,
        "thresholds": scanned,
        "fixed_0.50": fixed,
        "calibrated": calibrated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="load model weights before the configured stages")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device != "cuda":
        raise ValueError("only CUDA training is supported")
    device = _device()
    _seed(int(cfg["training"]["seed"]))
    assets_dir = Path(cfg["data"]["protocol_assets"])
    workspace = Path(cfg["data"]["root"])
    output = Path(cfg["data"]["output"])
    mode = str(cfg.get("experiment", {}).get("mode", "phase"))
    enable_phase = bool(cfg.get("model", {}).get("enable_phase_supervision", mode != "control"))
    enable_film = bool(cfg.get("model", {}).get("enable_film", mode == "full"))
    enable_feature_residual = bool(cfg.get("model", {}).get("enable_feature_residual", False))
    feature_residual_max = float(cfg.get("model", {}).get("feature_residual_max", 0.10))
    feature_residual_ratio_max = float(cfg.get("model", {}).get("feature_residual_ratio_max", 0.0))
    feature_residual_variant = str(cfg.get("model", {}).get("feature_residual_variant", "legacy"))
    enable_fused_inference = enable_film or enable_feature_residual
    output.mkdir(parents=True, exist_ok=True)
    if (output / "history.json").exists():
        raise FileExistsError(f"refusing to overwrite an existing training run: {output}")
    def progress(**fields):
        payload = {"updated_at": time.time(), **fields}
        temporary = output / "progress.tmp.json"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(output / "progress.json")
    progress(event="initializing")
    source_hashes = {p.name: _digest(p) for p in Path(__file__).parent.glob("*.py")}
    hashes = check_protocol(assets_dir, include_test=False)
    train_rows = load_rows(assets_dir / "train_validation500.csv")
    validation_rows = [row for row in train_rows if row["split"] == "validation"]
    gtdev_rows = load_rows(assets_dir / "gtdev60.csv")
    assets = USAAssets(workspace, assets_dir / "normalization_train1296.json")
    _validate_gtdev_assets(assets, gtdev_rows)
    train_loader = _loader(train_rows, assets, "train", cfg, shuffle=True)
    refresh_loader = _loader(train_rows, assets, "train", cfg, shuffle=False)
    model = AgriFMA4USA(cfg["data"]["agrifm_source"], cfg["data"]["pretrained"],
                        use_checkpointing=bool(cfg["model"].get("use_checkpointing", True)),
                        use_support_features=bool(cfg["model"].get("use_support_features", True)),
                        use_patch_metadata=bool(cfg["model"].get("use_patch_metadata", True)),
                        warmup_backbone=bool(cfg["model"].get("warmup_backbone", True)),
                        feature_residual_variant=feature_residual_variant,
                        reliability_input=str(cfg["model"].get("reliability_input", "a4_cube")),
                        num_experts=int(cfg["model"].get("num_experts", 4))).to(device)
    (output / "pretrained_load_report.json").write_text(json.dumps(model.classifier.load_report, indent=2), encoding="utf-8")
    init_payload = None
    if args.init_checkpoint is not None:
        init_payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        load_report = _load_init_checkpoint(model, init_payload)
        (output / "init_checkpoint.json").write_text(json.dumps({
            "path": str(args.init_checkpoint.resolve()),
            "sha256": _digest(args.init_checkpoint),
            "stage": init_payload.get("stage"),
            "epoch": init_payload.get("epoch"),
            "metrics": init_payload.get("metrics"),
            **load_report,
        }, indent=2), encoding="utf-8")
    bank: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    previous_stage = None
    global_epoch = int(cfg["training"].get("start_epoch", 0))
    resume_best = cfg["training"].get("resume_best_miou")
    best = (float(resume_best), global_epoch) if resume_best is not None else (-1.0, None)
    history: list[dict[str, Any]] = []
    for stage, epochs in cfg["training"]["stage_epochs"].items():
        epochs = int(epochs)
        if epochs <= 0:
            continue
        model.set_stage(stage)
        model.set_film_scale(0.0)
        model.set_feature_residual_scale(0.0)
        model.set_feature_residual_ratio_max(feature_residual_ratio_max)
        progress(event="stage_start", stage=stage, epoch=global_epoch + 1,
                 trainable_parameters=[name for name, p in model.named_parameters() if p.requires_grad])
        optimizer = _optimizer(model, stage, cfg)
        reliability_stage = stage in {"M", "J", "B", "W", "C", "D"} and mode != "control"
        if reliability_stage and bank is None:
            progress(event="target_refresh", stage=stage, epoch=global_epoch + 1)
            bank, bank_report = _refresh_target_bank(model, refresh_loader, device, None,
                                                     cfg["training"])
            (output / "target_bank_initial.json").write_text(json.dumps(bank_report, indent=2), encoding="utf-8")
        for local_epoch in range(epochs):
            global_epoch += 1
            if reliability_stage and previous_stage is not None:
                progress(event="target_refresh", stage=stage, epoch=global_epoch)
                bank, bank_report = _refresh_target_bank(model, refresh_loader, device, bank,
                                                         cfg["training"])
                (output / f"target_bank_epoch_{global_epoch:03d}.json").write_text(json.dumps(bank_report, indent=2), encoding="utf-8")
            previous_stage = stage
            if stage == "J" and enable_film:
                model.set_film_scale(0.2 * min(1.0, (local_epoch + 1) /
                                               max(1, int(cfg["training"].get("film_ramp_epochs", 3)))))
            else:
                model.set_film_scale(0.0)
            if enable_feature_residual and stage in {"J", "C", "D"}:
                ramp_epochs = max(1, int(cfg["training"].get("feature_residual_ramp_epochs", 3)))
                model.set_feature_residual_scale(feature_residual_max * min(
                    1.0, (local_epoch + 1) / ramp_epochs
                ))
                ratio_schedule = cfg["training"].get("feature_residual_ratio_schedule")
                if ratio_schedule:
                    schedule = [float(value) for value in ratio_schedule]
                    index = min(local_epoch, len(schedule) - 1) if stage == "J" else len(schedule) - 1
                    model.set_feature_residual_ratio_max(schedule[index])
            else:
                model.set_feature_residual_scale(0.0)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running: list[float] = []
            report_sums: dict[str, float] = {}
            started = time.time()
            for step, batch in enumerate(train_loader, 1):
                moved = _move(batch, device)
                target = target_valid = None
                if reliability_stage:
                    target, target_valid = _target_batch(batch, bank or {}, device)
                autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["training"].get("precision") == "bf16")
                with autocast:
                    output_model = model(
                        moved["agrifm_cube"], moved["a4_cube"], doy=moved["doy"],
                        observed_support=moved["observed_support"],
                        pseudo_class=moved["pseudo_class"],
                        pseudo_class_valid=moved["pseudo_class_valid"],
                        pseudo_purity=moved.get("pseudo_purity"),
                        pseudo_valid_fraction=moved.get("pseudo_valid_fraction"),
                        window_chunk_size=int(cfg["model"].get("window_chunk_size", 1)),
                        reliability_enabled=reliability_stage,
                    )
                    ramp_epochs = max(1, int(cfg["training"].get("reliability_ramp_epochs", 1)))
                    weight_ramp = min(1.0, (local_epoch + 1) / ramp_epochs) if stage in {"J", "C", "D"} else 0.0
                    loss, report = stage_loss(stage, output_model, moved, target, target_valid,
                                              reliability_enabled=reliability_stage,
                                              lambda_reliability=float(cfg["training"].get("lambda_reliability", 1.0)),
                                              weight_floor=float(cfg["training"].get("reliability_weight_floor", 0.05)),
                                              weight_ramp=weight_ramp,
                                              phase_weight=float(cfg["training"].get("phase_loss_weight", 0.3)) if enable_phase else 0.0,
                                              use_reliability_weighted_ce=bool(cfg["training"].get("use_reliability_weighted_ce", True)))
                    loss = loss / int(cfg["training"].get("gradient_accumulation", 1))
                loss.backward()
                if step % int(cfg["training"].get("gradient_accumulation", 1)) == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running.append(report["loss"])
                for key, value in report.items():
                    report_sums[key] = report_sums.get(key, 0.0) + value
                if step % int(cfg["training"].get("log_every", 10)) == 0:
                    progress(event="training", stage=stage, epoch=global_epoch,
                             step=step, total_steps=len(train_loader), loss=float(np.mean(running[-10:])))
                    print(json.dumps({"stage": stage, "epoch": global_epoch, "step": step,
                                      "loss": float(np.mean(running[-10:])),
                                      "seconds": time.time() - started}), flush=True)
            progress(event="gtdev_evaluation", stage=stage, epoch=global_epoch)
            if cfg["evaluation"].get("threshold_policy", "fixed_0.50") == "fixed_0.50":
                validation = {"available": False, "reason": "fixed_0.50 protocol; Validation330 references incomplete", "thresholds": [], "fixed_0.50": None, "calibrated": None}
            else:
                validation = _validation_metrics(model, assets, validation_rows, device,
                                                reliability_enabled=enable_fused_inference)
            fixed_gtdev = _gtdev_metrics(model, assets, gtdev_rows, device, threshold=0.5,
                                         reliability_enabled=enable_fused_inference)
            calibrated_threshold = 0.5
            if validation.get("available") and validation.get("calibrated") is not None:
                calibrated_threshold = float(validation["calibrated"]["threshold"])
            calibrated_gtdev = fixed_gtdev if calibrated_threshold == 0.5 else _gtdev_metrics(
                model, assets, gtdev_rows, device, threshold=calibrated_threshold,
                reliability_enabled=enable_fused_inference)
            # The top-level metric is always the GTdev score used for selection;
            # Validation only supplies a fixed, recorded threshold calibration.
            metrics = dict(calibrated_gtdev)
            metrics["fixed_0.50"] = fixed_gtdev
            metrics["validation"] = validation
            metrics["calibrated_gtdev"] = calibrated_gtdev
            record = {"stage": stage, "epoch": global_epoch, **metrics,
                      "stage_epoch": local_epoch + 1,
                      "weight_ramp": weight_ramp,
                      "train_components": {key: value / len(running) for key, value in report_sums.items()},
                      "train_loss": float(np.mean(running)) if running else None}
            history.append(record)
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            checkpoint = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "stage": stage, "epoch": global_epoch, "metrics": record,
                "protocol_hashes": hashes, "config": cfg,
                "source_code_sha256": source_hashes,
                "init_checkpoint_sha256": _digest(args.init_checkpoint) if args.init_checkpoint else None,
            }
            torch.save(checkpoint, output / f"epoch_{global_epoch:03d}.pt")
            selection_metric = cfg["evaluation"].get("selection_metric", "miou")
            selection_value = float(metrics.get(selection_metric, metrics["miou"]))
            if selection_value > best[0]:
                best = (selection_value, global_epoch)
                torch.save(checkpoint, output / "best.pt")
                (output / "best.json").write_text(json.dumps({"epoch": global_epoch, **metrics}, indent=2), encoding="utf-8")
            print(json.dumps({"event": "epoch_complete", **record, "best_epoch": best[1]}), flush=True)
            progress(event="epoch_complete", stage=stage, epoch=global_epoch, best_epoch=best[1], miou=metrics["miou"])
    progress(event="completed", epoch=global_epoch, best_epoch=best[1])
    (output / "completed.json").write_text(json.dumps({"epochs": global_epoch, "best_epoch": best[1], "best_miou": best[0], "source_code_sha256": source_hashes}, indent=2), encoding="utf-8")
    print(json.dumps({"event": "completed", "best": best, "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
