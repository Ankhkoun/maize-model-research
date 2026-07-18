"""Learnable Mexican-hat wavelet position encoding for temporal tokens."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class LearnableWaveletPositionEncoding(nn.Module):
    """Inject bounded, learnable multi-scale wavelet responses into tokens."""

    def __init__(
        self,
        dim: int,
        scale_init_days: Sequence[float] = (7.0, 17.5, 35.0),
        scale_min_days: float = 3.5,
        scale_max_days: float = 35.0,
        shift_init_days: Sequence[float] = (0.0, 0.0, 0.0),
        shift_max_abs_days: float = 7.0,
        support_radius_days: float = 42.0,
        alpha_init: float = 0.01,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if len(scale_init_days) != 3:
            raise ValueError("E1 requires exactly three wavelet bases")
        if len(scale_init_days) != len(shift_init_days):
            raise ValueError("scale_init_days and shift_init_days must have equal length")
        if scale_min_days <= 0 or scale_max_days <= scale_min_days:
            raise ValueError("scale bounds must satisfy 0 < min < max")
        if shift_max_abs_days <= 0:
            raise ValueError("shift_max_abs_days must be positive")
        if support_radius_days <= 0:
            raise ValueError("support_radius_days must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")

        scale_init = torch.as_tensor(scale_init_days, dtype=torch.float32)
        shift_init = torch.as_tensor(shift_init_days, dtype=torch.float32)
        if torch.any(scale_init < scale_min_days) or torch.any(scale_init > scale_max_days):
            raise ValueError("initial scales must lie inside the configured bounds")
        if torch.any(shift_init.abs() > shift_max_abs_days):
            raise ValueError("initial shifts must lie inside the configured bounds")

        self.dim = int(dim)
        self.num_bases = len(scale_init_days)
        self.scale_min_days = float(scale_min_days)
        self.scale_max_days = float(scale_max_days)
        self.shift_max_abs_days = float(shift_max_abs_days)
        self.support_radius_days = float(support_radius_days)
        self.eps = float(eps)

        shift_fraction = (shift_init / shift_max_abs_days).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        self.raw_scales = nn.Parameter(scale_init.clone())
        self.raw_shifts = nn.Parameter(torch.atanh(shift_fraction))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.fusion = nn.Linear(self.num_bases * self.dim, self.dim, bias=False)

    @property
    def scales(self) -> torch.Tensor:
        return self.raw_scales.clamp(self.scale_min_days, self.scale_max_days)

    @property
    def shifts(self) -> torch.Tensor:
        return self.shift_max_abs_days * torch.tanh(self.raw_shifts)

    def _validate_inputs(
        self,
        tokens: torch.Tensor,
        doy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if tokens.ndim != 4:
            raise ValueError("tokens must have shape [B,N,T,D]")
        if doy.ndim != 2:
            raise ValueError("doy must have shape [B,T]")
        if valid_mask.ndim != 2 or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be a boolean tensor with shape [B,T]")
        batch, _, time, dim = tokens.shape
        if dim != self.dim:
            raise ValueError(f"tokens feature dimension must equal {self.dim}")
        if doy.shape != (batch, time):
            raise ValueError("doy shape must match the tokens batch and time dimensions")
        if valid_mask.shape != (batch, time):
            raise ValueError("valid_mask shape must match the tokens batch and time dimensions")
        if not tokens.is_floating_point() or not doy.is_floating_point():
            raise ValueError("tokens and doy must be floating-point tensors")
        if tokens.device != doy.device or tokens.device != valid_mask.device:
            raise ValueError("tokens, doy, and valid_mask must be on the same device")
        if valid_mask.any() and not torch.isfinite(doy[valid_mask]).all():
            raise ValueError("doy must be finite at valid frames")

    def forward(
        self,
        tokens: torch.Tensor,
        doy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(tokens, doy, valid_mask)

        safe_doy = torch.where(valid_mask, doy, torch.zeros_like(doy))
        delta_days = safe_doy[:, :, None] - safe_doy[:, None, :]

        scales = self.scales.float()
        shifts = self.shifts.float()
        centered_delta = delta_days[:, None].float() - shifts[None, :, None, None]
        support = centered_delta.abs() <= self.support_radius_days
        key_mask = valid_mask[:, None, None, :]
        query_mask = valid_mask[:, None, :, None]
        active = support & key_mask & query_mask
        safe_centered_delta = torch.where(
            active,
            centered_delta,
            torch.zeros_like(centered_delta),
        )
        u = safe_centered_delta / scales[None, :, None, None]
        weights = scales.rsqrt()[None, :, None, None] * (1.0 - u.square())
        weights = weights * torch.exp(-0.5 * u.square())
        weights = torch.where(active, weights, torch.zeros_like(weights))
        denominator = weights.abs().sum(dim=-1, keepdim=True).clamp_min(self.eps)
        normalized_weights = (weights / denominator).to(dtype=tokens.dtype)

        safe_tokens = torch.where(
            valid_mask[:, None, :, None],
            tokens,
            torch.zeros_like(tokens),
        )
        responses = torch.einsum("brij,bnjd->bnrid", normalized_weights, safe_tokens)
        batch, patches, _, time, dim = responses.shape
        fused_input = responses.permute(0, 1, 3, 2, 4).reshape(
            batch, patches, time, self.num_bases * dim
        )
        wavelet_residual = self.fusion(fused_input)
        wavelet_residual = wavelet_residual * valid_mask[:, None, :, None]
        return tokens + self.alpha.to(tokens.dtype) * wavelet_residual
