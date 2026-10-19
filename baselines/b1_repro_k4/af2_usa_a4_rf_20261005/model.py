"""New full-tile AgriFM classifier plus spatially aligned A4 reliability."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .agrifm import AgriFMClassifier
from .reliability import A4WindowReliability


class StageFiLMAdapter(nn.Module):
    """Map window-level stage context back to the 64x64 F1 feature grid."""

    def __init__(self, dim: int = 128, channels: int = 256) -> None:
        super().__init__()
        self.gamma = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, channels))
        self.beta = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, channels))
        # The adapter starts as an exact identity and is enabled by the J stage.
        self.gate = nn.Parameter(torch.tensor(-6.0))

    def forward(self, f1_groups: Tensor, stage_context: Tensor, scale: float) -> Tensor:
        if f1_groups.ndim != 5 or tuple(f1_groups.shape[1:]) != (4, 256, 64, 64):
            raise ValueError("F1 groups must be [B,4,256,64,64]")
        if stage_context.ndim != 4 or tuple(stage_context.shape[1:3]) != (16, 1024):
            raise ValueError("stage context must be [B,16,1024,D]")
        batch, _, _, dim = stage_context.shape
        grid = stage_context.reshape(batch, 4, 4, 32, 32, dim)
        grid = grid.permute(0, 1, 3, 2, 4, 5).reshape(batch, 128, 128, dim)
        grid = grid.permute(0, 3, 1, 2).contiguous()
        grid = F.avg_pool2d(grid, kernel_size=2, stride=2)
        pooled = grid.permute(0, 2, 3, 1).contiguous()
        gamma = self.gamma(pooled).permute(0, 3, 1, 2).unsqueeze(1)
        beta = self.beta(pooled).permute(0, 3, 1, 2).unsqueeze(1)
        strength = float(scale) * torch.sigmoid(self.gate)
        return f1_groups * (1.0 + strength * torch.tanh(gamma)) + strength * beta


class ExpertFeatureResidualAdapter(nn.Module):
    """Map routed K=4 expert features back to the four F1 temporal groups."""

    def __init__(self, dim: int = 128, channels: int = 256, variant: str = "legacy"):
        super().__init__()
        self.variant = str(variant)
        if self.variant == "legacy":
            self.projection = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, channels),
                nn.GELU(),
                nn.Linear(channels, channels),
            )
            # Preserve the currently deployed residual implementation.
            nn.init.normal_(self.projection[-1].weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(self.projection[-1].bias)
        elif self.variant == "raw_bounded":
            self.projection = nn.Linear(dim, channels)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        elif self.variant == "ln_bounded":
            self.projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, channels))
            nn.init.zeros_(self.projection[-1].weight)
            nn.init.zeros_(self.projection[-1].bias)
        else:
            raise ValueError(f"unsupported feature residual variant: {self.variant}")

    def forward_with_stats(self, f1_groups: Tensor, expert_features: Tensor,
                           scale: float, max_ratio: float = 0.0) -> tuple[Tensor, dict[str, Tensor]]:
        if f1_groups.ndim != 5 or tuple(f1_groups.shape[1:]) != (4, 256, 64, 64):
            raise ValueError("F1 groups must be [B,4,256,64,64]")
        if expert_features.ndim != 5 or tuple(expert_features.shape[1:3]) != (16, 1024):
            raise ValueError("expert features must be [B,16,1024,4,D]")
        batch, _, _, groups, dim = expert_features.shape
        if groups != 4:
            raise ValueError("expert features must contain four F1 temporal groups")
        # [window_row, window_col, patch_row, patch_col, group, dim]
        grid = expert_features.reshape(batch, 4, 4, 32, 32, groups, dim)
        grid = grid.permute(0, 5, 6, 1, 3, 2, 4).contiguous()
        grid = grid.reshape(batch, groups, dim, 128, 128)
        grid = F.avg_pool2d(grid.reshape(batch * groups, dim, 128, 128), 2, 2)
        grid = grid.reshape(batch, groups, dim, 64, 64).permute(0, 1, 3, 4, 2)
        residual = self.projection(grid).permute(0, 1, 4, 2, 3)
        applied = float(scale) * residual
        f1_norm = torch.linalg.vector_norm(f1_groups.float().flatten(1), dim=1)
        raw_residual_norm = torch.linalg.vector_norm(applied.float().flatten(1), dim=1)
        raw_ratio = raw_residual_norm / f1_norm.clamp_min(1.0e-12)
        if max_ratio > 0.0:
            factor = (float(max_ratio) / raw_ratio.clamp_min(1.0e-12)).clamp_max(1.0).detach()
            applied = applied * factor.view(-1, 1, 1, 1, 1).to(applied.dtype)
        residual_norm = torch.linalg.vector_norm(applied.float().flatten(1), dim=1)
        ratio = residual_norm / f1_norm.clamp_min(1.0e-12)
        stats = {
            "residual_ratio_mean": ratio.mean(),
            "residual_ratio_p95": torch.quantile(ratio, 0.95),
            "residual_norm": residual_norm.mean(),
            "f1_norm": f1_norm.mean(),
        }
        return f1_groups + applied, stats

    def forward(self, f1_groups: Tensor, expert_features: Tensor, scale: float) -> Tensor:
        return self.forward_with_stats(f1_groups, expert_features, scale)[0]


class MultiScaleReliabilityAdapter(nn.Module):
    """Convert AgriFM's native four temporal groups and F1/F2/F3 into A4 tokens."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.f1 = nn.Conv2d(256, dim, 1)
        self.f2 = nn.Conv2d(256, dim, 1)
        self.f3 = nn.Conv2d(256, dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.group_scale = nn.Parameter(torch.zeros(3))

    def forward(self, f1: Tensor, f2: Tensor, f3: Tensor) -> Tensor:
        batch = f1.shape[0]
        groups = f1.reshape(batch, 4, 256, 64, 64)
        def project(feature: Tensor, layer: nn.Conv2d, size: tuple[int, int]) -> Tensor:
            if feature.shape[1] != 1024:
                raise ValueError("AgriFM multiscale features must have 1024 channels")
            feature = feature.reshape(batch * 4, 256, feature.shape[-2], feature.shape[-1])
            feature = layer(feature)
            return F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
        p1 = F.interpolate(self.f1(groups.reshape(batch * 4, 256, 64, 64)),
                           size=(128, 128), mode="bilinear", align_corners=False)
        p2 = project(f2, self.f2, (128, 128))
        p3 = project(f3, self.f3, (128, 128))
        merged = p1 + self.group_scale[1] * p2 + self.group_scale[2] * p3
        merged = self.norm(merged.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        merged = merged.reshape(batch, 4, 128, 128, 128)
        windows = merged.reshape(batch, 4, 128, 4, 32, 4, 32)
        windows = windows.permute(0, 3, 5, 4, 6, 1, 2).contiguous()
        return windows.reshape(batch, 16, 1024, 4, 128)


class AgriFMA4USA(nn.Module):
    def __init__(self, agrifm_source: str | Path, pretrained: str | Path,
                 *, use_checkpointing: bool = True,
                 use_support_features: bool = True,
                 use_patch_metadata: bool = True, warmup_backbone: bool = True,
                 feature_residual_variant: str = "legacy",
                 reliability_input: str = "a4_cube",
                 num_experts: int = 4):
        super().__init__()
        self.classifier = AgriFMClassifier(
            agrifm_source, pretrained, use_checkpointing=use_checkpointing,
        )
        self.reliability = A4WindowReliability(
            use_support_features=use_support_features,
            use_patch_metadata=use_patch_metadata,
            num_experts=num_experts,
        )
        self.stage_film = StageFiLMAdapter()
        self.expert_feature_residual = ExpertFeatureResidualAdapter(
            variant=feature_residual_variant
        )
        self.multiscale_reliability = MultiScaleReliabilityAdapter()
        self.reliability_input = str(reliability_input)
        self.stage = "A"
        self.warmup_backbone = bool(warmup_backbone)
        self.film_scale = 0.0
        self.feature_residual_scale = 0.0
        self.feature_residual_ratio_max = 0.0
        self.set_stage("A")

    def set_stage(self, stage: str) -> dict[str, int]:
        if stage not in {"A", "M", "J", "B", "W", "C", "D"}:
            raise ValueError(f"unsupported training stage: {stage}")
        self.stage = stage
        for parameter in self.classifier.parameters():
            parameter.requires_grad_(False)
        for parameter in self.classifier.decoder.parameters():
            parameter.requires_grad_(True)
        for parameter in self.reliability.parameters():
            parameter.requires_grad_(stage in {"M", "J", "B", "W", "C", "D"})
        for parameter in self.multiscale_reliability.parameters():
            parameter.requires_grad_(stage in {"M", "J", "B", "W", "C", "D"})
        for parameter in self.stage_film.parameters():
            parameter.requires_grad_(False)
        for parameter in self.expert_feature_residual.parameters():
            parameter.requires_grad_(stage in {"J", "W", "C", "D"})
        if stage == "W" and self.warmup_backbone:
            for parameter in self.classifier.patch_embed.parameters():
                parameter.requires_grad_(True)
            for parameter in self.classifier.backbone.layers[0].parameters():
                parameter.requires_grad_(True)
        if stage == "D":
            for parameter in self.classifier.backbone.layers[3].parameters():
                parameter.requires_grad_(True)
        return {
            "decoder": sum(p.numel() for p in self.classifier.decoder.parameters() if p.requires_grad),
            "last_swin_stage": sum(p.numel() for p in self.classifier.backbone.layers[3].parameters() if p.requires_grad),
            "reliability": sum(p.numel() for p in self.reliability.parameters() if p.requires_grad),
            "multiscale_reliability": sum(p.numel() for p in self.multiscale_reliability.parameters() if p.requires_grad),
            "stage_film": sum(p.numel() for p in self.stage_film.parameters() if p.requires_grad),
            "expert_feature_residual": sum(p.numel() for p in self.expert_feature_residual.parameters() if p.requires_grad),
        }

    def set_film_scale(self, scale: float) -> None:
        if not 0.0 <= float(scale) <= 0.2:
            raise ValueError("FiLM scale must be in [0,0.2]")
        self.film_scale = float(scale)

    def set_feature_residual_scale(self, scale: float) -> None:
        if not 0.0 <= float(scale) <= 0.2:
            raise ValueError("feature residual scale must be in [0,0.2]")
        self.feature_residual_scale = float(scale)

    def set_feature_residual_ratio_max(self, ratio: float) -> None:
        if float(ratio) < 0.0:
            raise ValueError("feature residual ratio must be nonnegative")
        self.feature_residual_ratio_max = float(ratio)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.stage not in {"W", "D"} or (self.stage == "W" and not self.warmup_backbone):
            self.classifier.backbone.eval()
            self.classifier.patch_embed.eval()
        else:
            for index, layer in enumerate(self.classifier.backbone.layers):
                layer.train(mode if index == (0 if self.stage == "W" else 3) else False)
            self.classifier.patch_embed.train(mode if self.stage == "W" else False)
        return self

    def forward(self, agrifm_cube: Tensor, a4_cube: Tensor | None, *, doy: Tensor,
                observed_support: Tensor, pseudo_class: Tensor,
                pseudo_class_valid: Tensor, pseudo_purity: Tensor | None = None,
                pseudo_valid_fraction: Tensor | None = None,
                window_chunk_size: int = 1,
                reliability_enabled: bool = True) -> dict[str, Tensor]:
        # AgriFMClassifier already limits no-grad to its frozen encoder. The
        # decoder must remain attached to autograd during Stage A.
        f1, f2, f3 = self.classifier.encode_features(agrifm_cube)
        f1_groups = self.classifier._f1_groups(f1)
        if not reliability_enabled:
            batch = agrifm_cube.shape[0]
            logits = self.classifier.decode_features(f1, f2, f3)
            reliability = logits.new_ones(batch, 16, 1024)
            return {"logits": logits, "reliability": reliability,
                    "reliability_map": logits.new_ones(batch, 256, 256),
                    "residual_ratio_mean": logits.new_zeros(()),
                    "residual_ratio_p95": logits.new_zeros(()),
                    "residual_norm": logits.new_zeros(()),
                    "f1_norm": torch.linalg.vector_norm(f1.float().flatten(1), dim=1).mean()}
        if self.reliability_input == "multiscale_agrifm":
            shared_tokens = self.multiscale_reliability(f1, f2, f3)
            bins = ((0, 7), (7, 14), (14, 20), (20, 26))
            shared_doy = torch.stack([doy[:, start:end].float().mean(1)
                                      for start, end in bins], dim=1)
            shared_support = torch.stack([observed_support[..., start:end].float().mean(-1)
                                          for start, end in bins], dim=-1)
            rel = self.reliability.forward_from_tokens(
                shared_tokens, doy=shared_doy, observed_support=shared_support,
                pseudo_class=pseudo_class, pseudo_class_valid=pseudo_class_valid,
                pseudo_purity=pseudo_purity, pseudo_valid_fraction=pseudo_valid_fraction,
                window_chunk_size=window_chunk_size,
            )
        else:
            if a4_cube is None:
                raise ValueError("a4_cube is required for reliability_input=a4_cube")
            rel = self.reliability(
                a4_cube, f1_groups, doy=doy, observed_support=observed_support,
                pseudo_class=pseudo_class, pseudo_class_valid=pseudo_class_valid,
                pseudo_purity=pseudo_purity, pseudo_valid_fraction=pseudo_valid_fraction,
                window_chunk_size=window_chunk_size,
            )
        adapted_f1_groups = f1_groups
        residual_stats = {
            "residual_ratio_mean": f1_groups.new_zeros(()),
            "residual_ratio_p95": f1_groups.new_zeros(()),
            "residual_norm": f1_groups.new_zeros(()),
            "f1_norm": torch.linalg.vector_norm(f1_groups.float().flatten(1), dim=1).mean(),
        }
        if self.film_scale > 0.0:
            adapted_f1_groups = self.stage_film(adapted_f1_groups, rel["stage_context"], self.film_scale)
        if self.feature_residual_scale > 0.0:
            adapted_f1_groups, residual_stats = self.expert_feature_residual.forward_with_stats(
                adapted_f1_groups, rel["fused_expert_features"], self.feature_residual_scale,
                self.feature_residual_ratio_max,
            )
        adapted_f1 = adapted_f1_groups.permute(0, 2, 1, 3, 4).contiguous().reshape(
            f1.shape[0], 1024, 64, 64
        )
        logits = self.classifier.decode_features(adapted_f1, f2, f3)
        return {"logits": logits, **rel, **residual_stats}


__all__ = ["AgriFMA4USA", "StageFiLMAdapter", "ExpertFeatureResidualAdapter"]
