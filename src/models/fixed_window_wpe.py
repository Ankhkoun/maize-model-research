"""Content-only fixed-window Mexican-hat wavelet position encoding."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FixedWindowMexicanHatWaveletEncoding(nn.Module):
    """Fuse three fixed temporal wavelet windows into a bounded residual."""

    def __init__(
        self,
        dim: int,
        radii: Sequence[int] = (2, 6, 12),
        sigmas: Sequence[float] = (1.0, 3.0, 6.0),
        shifts: Sequence[float] = (0.0, 0.0, 0.0),
        gamma_init: float = 0.02,
        gamma_parameterization: str = "positive_bounded",
        gamma_max: float = 0.20,
        eps: float = 1.0e-6,
        boundary_mode: str = "masked_renormalized",
        force_fp32: bool = True,
    ) -> None:
        super().__init__()
        radii_tuple = tuple(map(int, radii))
        sigmas_tuple = tuple(map(float, sigmas))
        shifts_tuple = tuple(map(float, shifts))
        if dim <= 0 or len(radii_tuple) != 3 or any(radius <= 0 for radius in radii_tuple):
            raise ValueError("fixed-window WPE requires positive dim and three positive radii")
        if len(sigmas_tuple) != 3 or any(sigma <= 0.0 for sigma in sigmas_tuple):
            raise ValueError("fixed-window WPE requires three positive sigmas")
        if len(shifts_tuple) != 3 or shifts_tuple != (0.0, 0.0, 0.0):
            raise ValueError("fixed-window WPE shifts are fixed at zero")
        if gamma_parameterization not in {"positive_bounded", "fixed_positive"}:
            raise ValueError("unsupported fixed-window gamma parameterization")
        if not 0.0 < gamma_init < gamma_max:
            raise ValueError("gamma_init must lie strictly between zero and gamma_max")
        if boundary_mode != "masked_renormalized":
            raise ValueError("fixed-window WPE requires masked-renormalized boundaries")

        self.dim = int(dim)
        self.eps = float(eps)
        self.gamma_max = float(gamma_max)
        self.window_names = tuple(f"K{2 * radius + 1}" for radius in radii_tuple)
        self.gamma_parameterization = gamma_parameterization
        self.boundary_mode = boundary_mode
        self.force_fp32 = bool(force_fp32)
        self.raw_gates = nn.Parameter(torch.zeros(3, dim))
        if gamma_parameterization == "positive_bounded":
            ratio = torch.tensor(gamma_init / gamma_max)
            self.raw_gamma = nn.Parameter(torch.logit(ratio))
            self.register_buffer("fixed_gamma", None)
        else:
            self.register_parameter("raw_gamma", None)
            self.register_buffer("fixed_gamma", torch.tensor(gamma_init))

        self.input_norm = nn.LayerNorm(dim)
        self.output_norm = nn.LayerNorm(dim)
        self.register_buffer("radii", torch.tensor(radii_tuple, dtype=torch.long))
        self.register_buffer("sigmas", torch.tensor(sigmas_tuple, dtype=torch.float32))
        self.register_buffer("shifts", torch.tensor(shifts_tuple, dtype=torch.float32))
        for index, (radius, sigma) in enumerate(
            zip(radii_tuple, sigmas_tuple, strict=True)
        ):
            offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
            u = offsets / sigma
            kernel = (1.0 - u.square()) * torch.exp(-0.5 * u.square())
            kernel = kernel - kernel.mean()
            kernel = kernel / kernel.abs().sum().clamp_min(self.eps)
            self.register_buffer(f"offsets_{index}", offsets)
            self.register_buffer(f"kernel_{index}", kernel)

        self.register_buffer(
            "diagnostic_residual_sq", torch.tensor(0.0), persistent=False
        )
        self.register_buffer(
            "diagnostic_input_sq", torch.tensor(0.0), persistent=False
        )

    @property
    def offsets(self) -> list[torch.Tensor]:
        return [getattr(self, f"offsets_{index}") for index in range(3)]

    @property
    def kernels(self) -> list[torch.Tensor]:
        return [getattr(self, f"kernel_{index}") for index in range(3)]

    @property
    def gamma_value(self) -> torch.Tensor:
        if self.raw_gamma is not None:
            return self.gamma_max * torch.sigmoid(self.raw_gamma)
        assert self.fixed_gamma is not None
        return self.fixed_gamma

    def fusion_weights(self) -> torch.Tensor:
        """Return per-channel weights with shape ``[3, D]``."""
        return torch.softmax(self.raw_gates, dim=0)

    def reset_diagnostics(self) -> None:
        self.diagnostic_residual_sq.zero_()
        self.diagnostic_input_sq.zero_()

    def diagnostics(self) -> dict[str, object]:
        weights = self.fusion_weights().detach().float().cpu()
        denominator = float(self.diagnostic_input_sq.detach().cpu())
        ratio = 0.0
        if denominator > 0.0:
            ratio = (
                float(self.diagnostic_residual_sq.detach().cpu()) / denominator
            ) ** 0.5
        radii = [int(radius) for radius in self.radii.detach().cpu().tolist()]
        return {
            "effective_residual_ratio": ratio,
            "gamma": float(self.gamma_value.detach().cpu()),
            "fusion": {
                name: {
                    "mean": float(row.mean()),
                    "min": float(row.min()),
                    "max": float(row.max()),
                }
                for name, row in zip(self.window_names, weights, strict=True)
            },
            "windows": [2 * radius + 1 for radius in radii],
            "radii": radii,
            "sigmas": [float(value) for value in self.sigmas.detach().cpu().tolist()],
            "shifts": [float(value) for value in self.shifts.detach().cpu().tolist()],
            "boundary_mode": self.boundary_mode,
            "dtype": "float32",
        }

    def _response(
        self, safe: torch.Tensor, valid_mask: torch.Tensor, basis: int
    ) -> torch.Tensor:
        time = valid_mask.shape[1]
        offsets = self.offsets[basis].to(valid_mask.device).long()
        raw_indices = torch.arange(time, device=valid_mask.device)[:, None] + offsets
        in_range = (raw_indices >= 0) & (raw_indices < time)
        indices = raw_indices.clamp(0, time - 1)
        active = in_range[None] & valid_mask[:, indices]
        raw = self.kernels[basis].to(safe.device)[None, None, :]
        count = active.sum(dim=-1, keepdim=True)
        active_mean = (raw * active).sum(dim=-1, keepdim=True) / count.clamp_min(1)
        centered = torch.where(active, raw - active_mean, torch.zeros_like(raw))
        weights = centered / centered.abs().sum(dim=-1, keepdim=True).clamp_min(
            self.eps
        )
        weights = torch.where(count >= 2, weights, torch.zeros_like(weights))
        return torch.einsum(
            "btk,bntkd->bntd", weights.to(safe.dtype), safe[:, :, indices, :]
        )

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if (
            tokens.ndim != 4
            or tokens.shape[-1] != self.dim
            or valid_mask.shape != (tokens.shape[0], tokens.shape[2])
            or valid_mask.dtype is not torch.bool
        ):
            raise ValueError("invalid fixed-window token or valid-mask contract")
        if self.force_fp32 and tokens.dtype != torch.float32:
            with torch.autocast(device_type=tokens.device.type, enabled=False):
                return self.forward(tokens.float(), valid_mask)

        normalized = self.input_norm(tokens)
        safe = torch.where(valid_mask[:, None, :, None], normalized, 0.0)
        responses = torch.stack(
            [self._response(safe, valid_mask, basis) for basis in range(3)], dim=2
        )
        residual = (
            responses * self.fusion_weights()[None, None, :, None, :]
        ).sum(dim=2)
        effective = (
            self.gamma_value
            * self.output_norm(residual)
            * valid_mask[:, None, :, None]
        )
        with torch.no_grad():
            mask = valid_mask[:, None, :, None]
            self.diagnostic_residual_sq.add_((effective.detach() * mask).square().sum())
            self.diagnostic_input_sq.add_((tokens.detach() * mask).square().sum())
        return tokens + effective
