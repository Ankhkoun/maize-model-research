"""A4-inspired K=4 phenology reliability head over 64px windows."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _initial_gaps(num_experts: int) -> Tensor:
    if num_experts == 4:
        centers = torch.tensor([0.15, 0.40, 0.65, 0.85])
    else:
        centers = (torch.arange(num_experts, dtype=torch.float32) + 0.5) / num_experts
    gaps = torch.cat((centers[:1], centers[1:] - centers[:-1], 1 - centers[-1:]))
    return gaps + torch.log((-torch.expm1(-gaps)).clamp_min(1e-8))


class A4WindowReliability(nn.Module):
    """Temporal reliability tokens, WPE evidence, AgriFM context, and K experts."""

    def __init__(self, channels: int = 10, dim: int = 128, frames: int = 26,
                 feature_dim: int = 128,
                 *, use_support_features: bool = True,
                 use_patch_metadata: bool = True, num_experts: int = 4):
        super().__init__()
        self.dim = dim
        self.feature_dim = int(feature_dim)
        self.use_support_features = bool(use_support_features)
        self.use_patch_metadata = bool(use_patch_metadata)
        self.num_experts = int(num_experts)
        if not 1 <= self.num_experts <= 6:
            raise ValueError("num_experts must be between 1 and 6")
        self.patch_projection = nn.Linear(channels * 4, dim)
        self.support_embedding = nn.Sequential(
            nn.Linear(1, 8), nn.GELU(), nn.Linear(8, dim),
        )
        self.slot_embedding = nn.Parameter(torch.zeros(1, frames, dim))
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)
        self.doy_embedding = nn.Sequential(
            nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=4 * dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_rel = nn.TransformerEncoder(layer, num_layers=1)
        self.temporal_norm = nn.LayerNorm(dim)
        self.progress = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim // 2),
                                      nn.GELU(), nn.Linear(dim // 2, 1))
        self.raw_gaps = nn.Parameter(_initial_gaps(self.num_experts))
        self.wpe_projection = nn.Linear(3 * dim, dim)
        nn.init.zeros_(self.wpe_projection.weight)
        nn.init.zeros_(self.wpe_projection.bias)
        self.wpe_gate = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(2 * dim + 1), nn.Linear(2 * dim + 1, 1))
            for _ in range(self.num_experts)
        ])
        self.context_projection = nn.Conv3d(256, dim, kernel_size=1)
        self.context_queries = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(self.num_experts)])
        self.pseudo_embedding = nn.Embedding(2, 8)
        self.patch_metadata = nn.Sequential(
            nn.LayerNorm(4), nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 16),
        )
        self.metadata_dim = 16 if self.use_patch_metadata else 0
        expert_dim = 2 * dim + 8 + self.metadata_dim
        self.experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(expert_dim), nn.Linear(expert_dim, dim),
                          nn.GELU(), nn.Linear(dim, 1))
            for _ in range(self.num_experts)
        ])
        # Each reliability expert also emits a feature vector.  This is
        # consumed by the F1 residual adapter; the scalar heads above remain
        # dedicated to delayed GMM/EMA reliability supervision.
        self.feature_experts = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                          nn.GELU(), nn.Linear(dim, self.feature_dim))
            for _ in range(self.num_experts)
        ])
        self.importance = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim // 2),
                          nn.GELU(), nn.Linear(dim // 2, 1))
            for _ in range(self.num_experts)
        ])

    def ordered_centers(self) -> Tensor:
        gaps = F.softplus(self.raw_gaps) + 1e-6
        cumulative = gaps.cumsum(0) / gaps.sum()
        return cumulative[:self.num_experts]

    def _context_windows(self, f1_groups: Tensor) -> Tensor:
        if f1_groups.ndim != 5 or tuple(f1_groups.shape[1:]) != (4, 256, 64, 64):
            raise ValueError(f"F1 groups must be [B,4,256,64,64], got {tuple(f1_groups.shape)}")
        batch = f1_groups.shape[0]
        projected = self.context_projection(f1_groups.permute(0, 2, 1, 3, 4))
        projected = projected.reshape(batch, self.dim, 4, 4, 16, 4, 16)
        projected = projected.permute(0, 3, 5, 2, 1, 4, 6).contiguous()
        projected = projected.reshape(batch * 16 * 4, self.dim, 16, 16)
        projected = F.interpolate(projected, size=(32, 32), mode="nearest")
        projected = projected.reshape(batch, 16, 4, self.dim, 32, 32)
        projected = projected.permute(0, 1, 4, 5, 2, 3).contiguous()
        return projected.reshape(batch, 16, 1024, 4, self.dim)

    @staticmethod
    def _window_patches(cube: Tensor, window_index: int) -> Tensor:
        row, col = divmod(window_index, 4)
        window = cube[:, :, :, row * 64:(row + 1) * 64, col * 64:(col + 1) * 64]
        patches = window.unfold(-2, 2, 2).unfold(-2, 2, 2)
        batch, frames, channels = patches.shape[:3]
        return patches.permute(0, 3, 4, 1, 2, 5, 6).contiguous().reshape(
            batch, 1024, frames, channels * 4
        )

    def _encode_window(self, cube: Tensor | None, context: Tensor, doy: Tensor,
                       support: Tensor, pseudo_class: Tensor,
                       pseudo_class_valid: Tensor, pseudo_purity: Tensor,
                       pseudo_valid_fraction: Tensor,
                       encoded_override: Tensor | None = None) -> dict[str, Tensor]:
        if encoded_override is None and cube is None:
            raise ValueError("cube or encoded_override is required")
        batch, patches, frames, _ = (encoded_override.shape if encoded_override is not None else cube.shape)
        if support.shape != (batch, patches, frames):
            raise ValueError(f"window observed_support must match encoded tokens, got {tuple(support.shape)}")
        support = support.float().clamp(0, 1)
        if encoded_override is None:
            calendar = doy.float() / 365.0
            doy_features = torch.stack((
                calendar,
                torch.sin(2.0 * math.pi * calendar),
                torch.cos(2.0 * math.pi * calendar),
                (calendar - 0.5).square(),
            ), dim=-1)
            x = self.patch_projection(cube) + self.slot_embedding[:, :frames] + self.doy_embedding(doy_features)[:, None]
            if self.use_support_features:
                x = x + self.support_embedding(support[..., None])
            x = x.reshape(batch * patches, frames, self.dim)
            observed = support.reshape(batch * patches, frames) > 0
            all_missing = ~observed.any(dim=1)
            safe_observed = observed.clone()
            safe_observed[all_missing, 0] = True
            encoded = self.temporal_rel(x, src_key_padding_mask=~safe_observed)
            encoded = self.temporal_norm(encoded).reshape(batch, patches, frames, self.dim)
        else:
            encoded = self.temporal_norm(encoded_override)
            all_missing = ~(support > 0).any(dim=-1)

        phi = self.progress(encoded).squeeze(-1).sigmoid()
        centers = self.ordered_centers()
        route = torch.softmax(-((phi[..., None] - centers) ** 2) / 0.08, dim=-1)
        stage_mass = route.sum(dim=2).clamp_min(1e-6)
        stages = torch.einsum("bptk,bptd->bpkd", route, encoded) / stage_mass[..., None]

        date_delta = (doy[:, :, None].float() - doy[:, None, :].float()).abs()
        scales = encoded.new_tensor([14.0, 42.0, 90.0])
        kernels = torch.exp(-date_delta[..., None] / scales)
        mass = torch.einsum("bpt,bqts->bpqs", support, kernels).clamp_min(1e-6)
        evidence = torch.einsum("bpt,bqts,bptd->bpqsd", support, kernels, encoded)
        evidence = evidence / mass[..., None]
        stage_evidence = torch.einsum("bptk,bptsd->bpksd", route, evidence)
        stage_evidence = stage_evidence.reshape(batch, patches, self.num_experts, 3 * self.dim) / stage_mass[..., None]
        projected = self.wpe_projection(stage_evidence)
        # ``kernels`` is [B, query_day, source_day, scale].  The source-day
        # sum must stay indexed by the query day when it is broadcast over
        # patches and query frames.  Using ``[:, None]`` here broadcasts the
        # query axis against the patch/query axes in the wrong order while
        # still passing the square 26x26 shape check.
        kernel_mass = kernels.sum(dim=2).clamp_min(1e-6)  # [B, query_day, scale]
        support_fraction = mass / kernel_mass[:, None, :, :]
        stage_support = torch.einsum("bptk,bpts->bpks", route, support_fraction)
        stage_support = (stage_support / stage_mass[..., None]).mean(dim=-1)
        alpha = torch.stack([
            torch.sigmoid(gate(torch.cat((stages[..., k, :], projected[..., k, :],
                                          stage_support[..., k, None]), dim=-1))).squeeze(-1)
            for k, gate in enumerate(self.wpe_gate)
        ], dim=-1)
        stages = stages + alpha[..., None] * projected

        # Each expert attends over all four aligned AgriFM context groups.
        context_vectors = context
        context_by_stage = []
        for k, query_layer in enumerate(self.context_queries):
            query = query_layer(stages[..., k, :])
            scores = (context_vectors * query[:, :, None, :]).sum(-1) / math.sqrt(self.dim)
            weights = scores.softmax(-1)
            context_by_stage.append((context_vectors * weights[..., None]).sum(-2))
        context_by_stage = torch.stack(context_by_stage, dim=2)
        embedded_class = self.pseudo_embedding(pseudo_class.long().clamp(0, 1))
        embedded_class = embedded_class * pseudo_class_valid.to(embedded_class.dtype)[..., None]
        if self.use_patch_metadata:
            support_summary = support.mean(dim=-1)
            metadata = torch.stack((support_summary, support.std(dim=-1),
                                    pseudo_purity.float().clamp(0, 1),
                                    pseudo_valid_fraction.float().clamp(0, 1)), dim=-1)
            metadata = self.patch_metadata(metadata)
        else:
            metadata = stages.new_zeros(batch, patches, 16)
        expert_values = torch.stack([
            torch.sigmoid(head(torch.cat(
                (stages[..., k, :], context_by_stage[..., k, :], embedded_class)
                if not self.use_patch_metadata else
                (stages[..., k, :], context_by_stage[..., k, :], embedded_class, metadata),
                dim=-1))).squeeze(-1)
            for k, head in enumerate(self.experts)
        ], dim=-1)
        feature_values = torch.stack([
            head(torch.cat(
                (stages[..., k, None, :].expand(-1, -1, context_vectors.shape[2], -1),
                 context_vectors),
                dim=-1))
            for k, head in enumerate(self.feature_experts)
        ], dim=2)
        importance = torch.stack([
            head(torch.cat((stages[..., k, :], context_by_stage[..., k, :]), dim=-1)).squeeze(-1)
            for k, head in enumerate(self.importance)
        ], dim=-1).softmax(-1)
        reliability = (importance * expert_values).sum(-1)
        reliability = torch.where(all_missing.reshape(batch, patches),
                                  torch.zeros_like(reliability), reliability)
        stage_features = stages
        stage_context = (importance[..., None] * (stage_features + context_by_stage)).sum(dim=2)
        fused_expert_features = (importance[..., None, None] * feature_values).sum(dim=2)
        fused_expert_features = fused_expert_features.masked_fill(
            all_missing.reshape(batch, patches, 1, 1), 0.0
        )
        return {"reliability": reliability, "phi": phi, "route": route,
                "importance": importance, "centers": centers,
                "wpe_gate": alpha, "stage_support": stage_support,
                "stage_features": stage_features, "stage_context": stage_context,
                "expert_features": feature_values,
                "fused_expert_features": fused_expert_features}

    def forward_from_tokens(self, tokens: Tensor, *, doy: Tensor,
                            observed_support: Tensor, pseudo_class: Tensor,
                            pseudo_class_valid: Tensor, pseudo_purity: Tensor | None = None,
                            pseudo_valid_fraction: Tensor | None = None,
                            window_chunk_size: int = 1) -> dict[str, Tensor]:
        """Run the A4 route/MoE on AgriFM-native temporal groups."""
        if tokens.ndim != 5 or tuple(tokens.shape[1:]) != (16, 1024, 4, self.dim):
            raise ValueError("shared tokens must be [B,16,1024,4,128]")
        batch = tokens.shape[0]
        if doy.shape != (batch, 4) or observed_support.shape != (batch, 16, 1024, 4):
            raise ValueError("shared DOY/support must be [B,4] and [B,16,1024,4]")
        if pseudo_class.shape != (batch, 16, 1024) or pseudo_class_valid.shape != (batch, 16, 1024):
            raise ValueError("patch class and its validity must be [B,16,1024]")
        if pseudo_purity is None:
            pseudo_purity = pseudo_class_valid.float()
        if pseudo_valid_fraction is None:
            pseudo_valid_fraction = pseudo_class_valid.float()
        outputs = []
        for window_index in range(16):
            outputs.append(self._encode_window(
                None, self._context_windows_from_tokens(tokens[:, window_index]), doy,
                observed_support[:, window_index], pseudo_class[:, window_index],
                pseudo_class_valid[:, window_index], pseudo_purity[:, window_index],
                pseudo_valid_fraction[:, window_index],
                encoded_override=tokens[:, window_index],
            ))
        result = {key: torch.stack([item[key] for item in outputs], dim=1)
                  for key in ("reliability", "phi", "route", "importance", "stage_support",
                              "stage_features", "stage_context", "expert_features",
                              "fused_expert_features")}
        result["centers"] = outputs[0]["centers"]
        result["wpe_gate"] = torch.stack([item["wpe_gate"] for item in outputs], dim=1)
        result["reliability_map"] = self.to_tile_map(result["reliability"])
        return result

    @staticmethod
    def _context_windows_from_tokens(tokens: Tensor) -> Tensor:
        # The shared AgriFM tokens already carry one vector per temporal group.
        return tokens

    def forward(self, a4_cube: Tensor, f1_groups: Tensor, *, doy: Tensor,
                observed_support: Tensor, pseudo_class: Tensor,
                pseudo_class_valid: Tensor, pseudo_purity: Tensor | None = None,
                pseudo_valid_fraction: Tensor | None = None,
                window_chunk_size: int = 1) -> dict[str, Tensor]:
        if a4_cube.ndim != 5 or tuple(a4_cube.shape[1:]) != (26, 10, 256, 256):
            raise ValueError("A4 cube must be normalized [B,26,10,256,256]")
        batch = a4_cube.shape[0]
        if doy.shape != (batch, 26):
            raise ValueError("DOY must be the 26 real calendar day-of-year values [B,26]")
        expected = (batch, 16, 1024, 26)
        if observed_support.shape != expected:
            raise ValueError(f"observed support must be {expected}")
        if pseudo_class.shape != (batch, 16, 1024) or pseudo_class_valid.shape != (batch, 16, 1024):
            raise ValueError("patch class and its validity must be [B,16,1024]")
        if pseudo_purity is None:
            pseudo_purity = pseudo_class_valid.float()
        if pseudo_valid_fraction is None:
            pseudo_valid_fraction = pseudo_class_valid.float()
        expected_patch = (batch, 16, 1024)
        if pseudo_purity.shape != expected_patch or pseudo_valid_fraction.shape != expected_patch:
            raise ValueError("patch purity and valid fraction must be [B,16,1024]")

        context = self._context_windows(f1_groups)
        outputs: list[dict[str, Tensor]] = []
        chunk_size = max(1, int(window_chunk_size))
        for first in range(0, 16, chunk_size):
            selected = range(first, min(first + chunk_size, 16))
            for window_index in selected:
                output = self._encode_window(
                    self._window_patches(a4_cube, window_index), context[:, window_index], doy,
                    observed_support[:, window_index], pseudo_class[:, window_index],
                    pseudo_class_valid[:, window_index], pseudo_purity[:, window_index],
                    pseudo_valid_fraction[:, window_index],
                )
                outputs.append(output)
        result = {}
        for key in ("reliability", "phi", "route", "importance", "stage_support",
                    "stage_features", "stage_context", "expert_features",
                    "fused_expert_features"):
            result[key] = torch.stack([item[key] for item in outputs], dim=1)
        result["centers"] = outputs[0]["centers"]
        result["wpe_gate"] = torch.stack([item["wpe_gate"] for item in outputs], dim=1)
        result["reliability_map"] = self.to_tile_map(result["reliability"])
        return result

    @staticmethod
    def to_tile_map(reliability: Tensor) -> Tensor:
        if reliability.ndim != 3 or reliability.shape[1:] != (16, 1024):
            raise ValueError("reliability must be [B,16,1024]")
        batch = reliability.shape[0]
        grid = reliability.reshape(batch, 4, 4, 32, 32)
        grid = grid.permute(0, 1, 3, 2, 4).reshape(batch, 128, 128)
        return grid.repeat_interleave(2, -2).repeat_interleave(2, -1)


__all__ = ["A4WindowReliability"]
