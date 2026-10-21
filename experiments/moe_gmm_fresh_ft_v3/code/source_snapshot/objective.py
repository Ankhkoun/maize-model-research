"""Stage-specific CE and delayed reliability objectives."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .data import windows_to_tile_grid


def classification_loss(logits: Tensor, target: Tensor, reliability_map: Tensor | None = None,
                        valid_pixel_mask: Tensor | None = None, *, weighted: bool = False,
                        weight_floor: float = 0.05) -> Tensor:
    if logits.ndim != 4 or target.shape != logits.shape[:1] + logits.shape[-2:]:
        raise ValueError("classification logits/target must be [B,2,H,W] and [B,H,W]")
    target = target.long()
    valid = target != 255 if valid_pixel_mask is None else valid_pixel_mask.bool() & (target != 255)
    safe_target = target.masked_fill(~valid, 255)
    pixel = F.cross_entropy(logits.float(), safe_target, ignore_index=255, reduction="none")
    if not weighted:
        return (pixel * valid).sum() / valid.sum().clamp_min(1)
    if reliability_map is None or reliability_map.shape != target.shape:
        raise ValueError("weighted CE requires an aligned reliability map")
    weight = (weight_floor + (1.0 - weight_floor) * torch.sigmoid(
        (reliability_map.detach().float() - 0.6) / 0.1
    )) * valid
    return (pixel * weight).sum() / weight.sum().clamp_min(1.0)


def reliability_supervision(output: dict[str, Tensor], target: Tensor | None,
                            target_valid: Tensor | None) -> tuple[Tensor, dict[str, Tensor]]:
    reliability = output["reliability"]
    zero = reliability.float().sum() * 0.0
    if target is None or target_valid is None or not bool(target_valid.any()):
        return zero, {"target_pixels": zero.detach(), "reliability_mse": zero.detach()}
    prediction_grid = windows_to_tile_grid(reliability.float())
    if target.shape != prediction_grid.shape or target_valid.shape != target.shape:
        raise ValueError("delayed target bank must be [B,128,128]")
    mask = target_valid.bool()
    mse = ((prediction_grid - target.detach().float()).square() * mask).sum() / mask.sum().clamp_min(1)
    importance = output["importance"].float().mean(dim=(0, 1, 2))
    balance = (importance - 1.0 / importance.numel()).square().sum()
    phi = output["phi"].float()
    support_monotonic = F.relu(phi[..., :-1] - phi[..., 1:]).mean()
    centers = output["centers"].float()
    min_separation = 0.15 if centers.numel() == 4 else 0.5 / centers.numel()
    separation = F.relu(min_separation - torch.diff(centers)).sum()
    aux = {"target_pixels": mask.sum().float(), "reliability_mse": mse,
           "importance_balance": balance, "phenology_monotonic": support_monotonic,
           "center_separation": separation}
    with torch.no_grad():
        prediction_values = prediction_grid[mask].detach().float()
        target_values = target[mask].detach().float()
        pred_centered = prediction_values - prediction_values.mean()
        target_centered = target_values - target_values.mean()
        denominator = pred_centered.square().sum().sqrt() * target_centered.square().sum().sqrt()
        aux["reliability_target_correlation"] = (pred_centered * target_centered).sum() / denominator.clamp_min(1e-12)
    return mse + 0.01 * balance + 0.02 * support_monotonic + 0.01 * separation, aux


def phase_route_supervision(output: dict[str, Tensor], batch: dict[str, Any]) -> tuple[Tensor, dict[str, Tensor]]:
    """Supervise route semantics with weak spectral phase targets.

    EMA/GMM remains the reliability target.  This separate objective only
    constrains the temporal expert route and is masked to high-confidence
    pseudo-corn patches by ``phase_valid``.
    """
    target = batch.get("phase_target")
    valid = batch.get("phase_valid")
    confidence = batch.get("phase_confidence")
    route = output.get("route")
    if route is None:
        anchor = output["logits"]
        zero = anchor.float().sum() * 0.0
        return zero, {"phase_pixels": zero.detach(), "phase_kl": zero.detach()}
    route = route.float()
    zero = route.sum() * 0.0
    if target is None or valid is None or confidence is None:
        return zero, {"phase_pixels": zero.detach(), "phase_kl": zero.detach()}
    target = target.to(route.device).float()
    valid = valid.to(route.device).bool()
    confidence = confidence.to(route.device).float().clamp(0.0, 1.0)
    if target.shape != route.shape or valid.shape != route.shape[:-1] or confidence.shape != valid.shape:
        raise ValueError(f"phase target shapes disagree: route={tuple(route.shape)} target={tuple(target.shape)}")
    mask = valid & (confidence > 0.0)
    if not bool(mask.any()):
        return zero, {"phase_pixels": zero.detach(), "phase_kl": zero.detach()}
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    log_route = route.clamp_min(1.0e-7).log()
    kl = (target * (target.clamp_min(1.0e-7).log() - log_route)).sum(dim=-1)
    weight = confidence * mask.to(confidence.dtype)
    phase_kl = (kl * weight).sum() / weight.sum().clamp_min(1.0)
    features = F.normalize(output["stage_features"].float(), dim=-1)
    similarity = torch.einsum("...kd,...ld->...kl", features, features)
    off_diagonal = similarity - torch.diag_embed(torch.diagonal(similarity, dim1=-2, dim2=-1))
    diversity = off_diagonal.square().sum() / max(1, off_diagonal.numel() - similarity.shape[-1] ** 2)
    usage = output["importance"].float().mean(dim=(0, 1, 2))
    balance = (usage - 1.0 / usage.numel()).square().sum()
    aux = {
        "phase_pixels": mask.sum().float(),
        "phase_kl": phase_kl,
        "phase_diversity": diversity,
        "phase_balance": balance,
        "phase_valid_fraction": mask.float().mean(),
    }
    return phase_kl + 0.02 * diversity + 0.01 * balance, aux


def stage_loss(stage: str, output: dict[str, Tensor], batch: dict[str, Any],
               target: Tensor | None, target_valid: Tensor | None,
               *, reliability_enabled: bool, lambda_reliability: float = 1.0,
               weight_floor: float = 0.05, weight_ramp: float = 1.0,
               phase_weight: float = 0.3,
               use_reliability_weighted_ce: bool = True) -> tuple[Tensor, dict[str, float]]:
    labels = batch["target"].to(output["logits"].device)
    valid = batch["valid_pixel_mask"].to(labels.device)
    ordinary = classification_loss(output["logits"], labels, valid_pixel_mask=valid)
    rel_loss, rel_metrics = reliability_supervision(output, target, target_valid) if reliability_enabled else (
        output["logits"].float().sum() * 0.0, {}
    )
    phase_loss, phase_metrics = phase_route_supervision(output, batch) if reliability_enabled and phase_weight > 0 else (
        output["logits"].float().sum() * 0.0, {}
    )
    if stage == "A":
        total = ordinary
        cls = ordinary
    elif stage in {"M", "B", "W"}:
        # Reliability learns from delayed targets while the decoder continues
        # ordinary pseudo-label training; no reliability weighting is active.
        total = ordinary + float(lambda_reliability) * rel_loss + float(phase_weight) * phase_loss
        cls = ordinary
    elif stage == "J":
        # Joint fusion first learns the expert feature residual with ordinary
        # CE; selective weighting is introduced only after this path settles.
        total = ordinary + float(lambda_reliability) * rel_loss + float(phase_weight) * phase_loss
        cls = ordinary
    elif stage in {"C", "D"} and use_reliability_weighted_ce:
        weighted = classification_loss(output["logits"], labels, output["reliability_map"],
                                       valid, weighted=True, weight_floor=weight_floor)
        cls = ordinary + float(weight_ramp) * (weighted - ordinary)
        total = cls + float(lambda_reliability) * rel_loss + float(phase_weight) * phase_loss
    elif stage in {"C", "D"}:
        # Safe ablation: keep the reliability branch supervised, but do not let
        # its prediction suppress classifier gradients on difficult pixels.
        total = ordinary + float(lambda_reliability) * rel_loss + float(phase_weight) * phase_loss
        cls = ordinary
    else:
        raise ValueError(f"unknown training stage {stage}")
    if not torch.isfinite(total):
        raise FloatingPointError(f"nonfinite Stage {stage} loss")
    report = {"loss": float(total.detach()), "classification_ce": float(cls.detach()),
              "ordinary_ce": float(ordinary.detach()), "reliability_loss": float(rel_loss.detach())}
    for key in ("residual_ratio_mean", "residual_ratio_p95", "residual_norm", "f1_norm"):
        value = output.get(key)
        if isinstance(value, Tensor):
            report[key] = float(value.detach())
    for key, value in rel_metrics.items():
        report[key] = float(value.detach())
    for key, value in phase_metrics.items():
        report[key] = float(value.detach())
    return total, report


__all__ = ["classification_loss", "reliability_supervision", "phase_route_supervision", "stage_loss"]
