"""Per-class GMM confidence targets with delayed predictions and EMA updates."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def _fit_two_gaussians(values: np.ndarray, *, steps: int = 40,
                      min_samples: int = 32, min_separation: float = 0.08,
                      min_weight: float = 0.03):
    values = np.asarray(values, dtype=np.float64)
    if values.size < min_samples or not np.isfinite(values).all() or float(values.std()) < 1e-7:
        return None
    q05, q95 = np.quantile(values, [0.05, 0.95])
    scaled = np.clip((values - q05) / max(float(q95 - q05), 1e-8), 0.0, 1.0)
    means = np.quantile(scaled, [0.25, 0.75])
    variances = np.full(2, max(float(scaled.var()), 1e-3))
    weights = np.full(2, 0.5)
    for _ in range(steps):
        logp = np.log(weights + 1e-12) - 0.5 * (np.log(2 * np.pi * variances) + (scaled[:, None] - means) ** 2 / variances)
        logp -= logp.max(axis=1, keepdims=True)
        posterior = np.exp(logp)
        posterior /= posterior.sum(axis=1, keepdims=True).clip(1e-12)
        mass = posterior.sum(axis=0).clip(1e-6)
        weights = mass / len(scaled)
        means = (posterior * scaled[:, None]).sum(axis=0) / mass
        variances = (posterior * (scaled[:, None] - means) ** 2).sum(axis=0) / mass
        variances = variances.clip(1e-5, 1.0)
    order = np.argsort(means)
    means, variances, weights = means[order], variances[order], weights[order]
    if means[1] - means[0] < min_separation or weights.min() < min_weight:
        return None
    all_scaled = np.clip((values - q05) / max(float(q95 - q05), 1e-8), 0.0, 1.0)
    logp = np.log(weights + 1e-12) - 0.5 * (np.log(2 * np.pi * variances) + (all_scaled[:, None] - means) ** 2 / variances)
    logp -= logp.max(axis=1, keepdims=True)
    posterior = np.exp(logp)
    posterior /= posterior.sum(axis=1, keepdims=True).clip(1e-12)
    return posterior[:, 0].astype(np.float32), "gmm", float(means[1] - means[0])


def build_targets(logits: Tensor, labels: Tensor, previous: Tensor | None = None,
                  previous_valid: Tensor | None = None, *, ema: float = 0.9,
                  target_mode: str = "gmm", gmm_steps: int = 40,
                  gmm_min_samples: int = 32, gmm_min_separation: float = 0.08,
                  gmm_min_weight: float = 0.03
                  ) -> tuple[Tensor, Tensor, dict[str, object]]:
    """Compute 2x2 loss targets; pseudo validity is a mask, never a target value."""
    if logits.ndim != 4 or logits.shape[1:] != (2, 256, 256):
        raise ValueError("target refresh expects full-tile [B,2,256,256] logits")
    batch = logits.shape[0]
    if labels.shape != (batch, 256, 256):
        raise ValueError(f"target refresh expects [{batch},256,256] pseudo labels")
    if previous is not None and previous.shape != (batch, 128, 128):
        raise ValueError(f"EMA target expects [{batch},128,128] previous targets")
    if previous_valid is not None and previous_valid.shape != (batch, 128, 128):
        raise ValueError(f"EMA target expects [{batch},128,128] previous validity")
    if (previous is None) != (previous_valid is None):
        raise ValueError("previous and previous_valid must be provided together")
    labels = labels.to(logits.device).long()
    valid_pixel = labels != 255
    per_pixel = F.cross_entropy(logits.float(), labels, ignore_index=255, reduction="none")
    valid_4d = valid_pixel.float().unsqueeze(1)
    count = F.avg_pool2d(valid_4d, 2, 2)[:, 0] * 4
    patch_loss = F.avg_pool2d((per_pixel * valid_pixel).unsqueeze(1), 2, 2)[:, 0] * 4
    patch_loss = patch_loss / count.clamp_min(1.0)
    positives = F.avg_pool2d((labels == 1).float().unsqueeze(1), 2, 2)[:, 0] * 4
    patch_class = (positives / count.clamp_min(1.0) >= 0.5).long()
    eligible = (count >= 3) & ((positives / count.clamp_min(1.0) >= 0.75) |
                               (positives / count.clamp_min(1.0) <= 0.25))

    target = torch.zeros_like(patch_loss)
    reports = []
    for batch_index in range(batch):
        sample_reports = []
        for class_id in (0, 1):
            mask = eligible[batch_index] & (patch_class[batch_index] == class_id)
            values = patch_loss[batch_index][mask].detach().cpu().numpy()
            if values.size == 0:
                sample_reports.append({"class": class_id, "mode": "empty", "count": 0})
                continue
            if target_mode not in {"gmm", "rank"}:
                raise ValueError(f"unsupported target mode: {target_mode}")
            fit = None if target_mode == "rank" else _fit_two_gaussians(
                values, steps=gmm_steps, min_samples=gmm_min_samples,
                min_separation=gmm_min_separation, min_weight=gmm_min_weight,
            )
            if fit is None:
                order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
                score = 1.0 - order / max(1, len(order) - 1)
                mode, separation = ("rank" if target_mode == "rank" else "rank_fallback"), None
            else:
                score, mode, separation = fit
            target[batch_index][mask] = torch.from_numpy(score).to(target)
            sample_reports.append({"class": class_id, "mode": mode, "count": int(values.size),
                                  "separation": separation})
        reports.append({
            "classes": sample_reports,
            "eligible_fraction": float(eligible[batch_index].float().mean()),
            "target_mean": float(target[batch_index][eligible[batch_index]].mean())
            if bool(eligible[batch_index].any()) else 0.0,
        })
    if previous is not None:
        old = eligible & previous_valid.bool()
        new = eligible & ~previous_valid.bool()
        target[old] = ema * previous.to(target)[old] + (1 - ema) * target[old]
        target[new] = target[new]
    report: dict[str, object] = {
        "classes": reports[0]["classes"] if batch == 1 else [item["classes"] for item in reports],
        "per_sample": reports,
        "eligible_fraction": float(eligible.float().mean()),
        "target_mean": float(target[eligible].mean()) if bool(eligible.any()) else 0.0,
    }
    return target.detach(), eligible.detach(), report


__all__ = ["build_targets"]
