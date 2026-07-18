"""Exact-style temporal-spatial ViT for direct semantic segmentation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .wavelet_position_encoding import LearnableWaveletPositionEncoding


class _TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feed_forward(self.feed_forward_norm(tokens))


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens, key_padding_mask=key_padding_mask)
        return self.norm(tokens)


class TSViTSegmentation(nn.Module):
    """Temporal-spatial Transformer that returns dense segmentation logits."""

    def __init__(self, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        required = (
            "image_size",
            "patch_size",
            "num_frames",
            "num_channels",
            "num_classes",
            "dim",
            "temporal_depth",
            "spatial_depth",
            "heads",
        )
        missing = [key for key in required if key not in model_config]
        if missing:
            raise ValueError(f"missing model configuration keys: {missing}")

        self.image_size = int(model_config["image_size"])
        self.patch_size = int(model_config["patch_size"])
        self.num_frames = int(model_config["num_frames"])
        self.num_channels = int(model_config["num_channels"])
        self.num_classes = int(model_config["num_classes"])
        self.dim = int(model_config["dim"])
        temporal_depth = int(model_config["temporal_depth"])
        spatial_depth = int(model_config["spatial_depth"])
        heads = int(model_config["heads"])
        mlp_ratio = float(model_config.get("mlp_ratio", 4.0))
        dropout = float(model_config.get("dropout", 0.0))
        embedding_dropout = float(model_config.get("emb_dropout", 0.0))

        positive_values = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_frames": self.num_frames,
            "num_channels": self.num_channels,
            "num_classes": self.num_classes,
            "dim": self.dim,
            "temporal_depth": temporal_depth,
            "spatial_depth": spatial_depth,
            "heads": heads,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= dropout < 1.0 or not 0.0 <= embedding_dropout < 1.0:
            raise ValueError("dropout values must lie in [0,1)")

        self.patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.patches_per_side**2
        patch_dim = self.num_channels * self.patch_size**2

        self.patch_embedding = nn.Linear(patch_dim, self.dim)
        self.doy_embedding = nn.Embedding(367, self.dim, padding_idx=0)
        self.temporal_class_tokens = nn.Parameter(
            torch.randn(1, self.num_classes, self.dim)
        )
        self.temporal_transformer = _TransformerEncoder(
            self.dim, temporal_depth, heads, mlp_ratio, dropout
        )
        self.spatial_position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, self.dim)
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.spatial_transformer = _TransformerEncoder(
            self.dim, spatial_depth, heads, mlp_ratio, dropout
        )
        self.segmentation_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_size**2),
        )

        wavelet_config = dict(model_config.get("wavelet", {}))
        wavelet_enabled = bool(wavelet_config.pop("enabled", False))
        self.wavelet: LearnableWaveletPositionEncoding | None
        if wavelet_enabled:
            self.wavelet = LearnableWaveletPositionEncoding(
                dim=self.dim,
                **wavelet_config,
            )
        else:
            self.wavelet = None

    def _validate_inputs(
        self,
        images: torch.Tensor,
        doy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        if images.ndim != 5:
            raise ValueError("images must have shape [B,T,C,H,W]")
        if not images.is_floating_point():
            raise ValueError("images must be a floating-point tensor")
        if doy.ndim != 2 or not doy.is_floating_point():
            raise ValueError("doy must be a floating-point tensor with shape [B,T]")
        if valid_mask.ndim != 2 or valid_mask.dtype is not torch.bool:
            raise ValueError("valid_mask must be a boolean tensor with shape [B,T]")

        batch, time, channels, height, width = images.shape
        if time != self.num_frames:
            raise ValueError(f"images time dimension must equal {self.num_frames}")
        if channels != self.num_channels:
            raise ValueError(f"images channel dimension must equal {self.num_channels}")
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"image height and width must both equal {self.image_size}"
            )
        if doy.shape != (batch, time):
            raise ValueError("doy shape must match the images batch and time dimensions")
        if valid_mask.shape != (batch, time):
            raise ValueError(
                "valid_mask shape must match the images batch and time dimensions"
            )
        if images.device != doy.device or images.device != valid_mask.device:
            raise ValueError("images, doy, and valid_mask must be on the same device")
        if valid_mask.any():
            if not torch.isfinite(doy[valid_mask]).all():
                raise ValueError("doy must be finite at valid frames")
            valid_images = images[valid_mask]
            if not torch.isfinite(valid_images).all():
                raise ValueError("images must be finite at valid frames")

    def forward(
        self,
        images: torch.Tensor,
        doy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(images, doy, valid_mask)
        batch, time, _, height, width = images.shape

        safe_images = torch.where(
            valid_mask[:, :, None, None, None],
            images,
            torch.zeros_like(images),
        )
        flattened_images = safe_images.reshape(
            batch * time, self.num_channels, height, width
        )
        patches = F.unfold(
            flattened_images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        patch_tokens = self.patch_embedding(patches)
        patch_tokens = patch_tokens.reshape(
            batch, time, self.num_patches, self.dim
        ).permute(0, 2, 1, 3)

        safe_doy = torch.where(valid_mask, doy, torch.zeros_like(doy))
        if valid_mask.any():
            valid_doy = safe_doy[valid_mask]
            is_integer = torch.isclose(
                valid_doy, valid_doy.round(), atol=1e-4, rtol=0.0
            )
            in_calendar = (valid_doy >= 1.0) & (valid_doy <= 366.0)
            if not (is_integer & in_calendar).all():
                raise ValueError(
                    "valid DOY values must be integer calendar days in [1,366]"
                )
        doy_indices = safe_doy.round().long()
        doy_tokens = self.doy_embedding(doy_indices).to(dtype=patch_tokens.dtype)
        temporal_tokens = patch_tokens + doy_tokens[:, None]
        if self.wavelet is not None:
            temporal_tokens = self.wavelet(temporal_tokens, doy, valid_mask)

        temporal_tokens = temporal_tokens.reshape(
            batch * self.num_patches, time, self.dim
        )
        class_tokens = self.temporal_class_tokens.expand(
            batch * self.num_patches, -1, -1
        )
        temporal_tokens = torch.cat((class_tokens, temporal_tokens), dim=1)

        time_padding = (~valid_mask)[:, None, :].expand(
            batch, self.num_patches, time
        ).reshape(batch * self.num_patches, time)
        class_padding = torch.zeros(
            batch * self.num_patches,
            self.num_classes,
            dtype=torch.bool,
            device=images.device,
        )
        temporal_padding = torch.cat((class_padding, time_padding), dim=1)
        temporal_output = self.temporal_transformer(
            temporal_tokens,
            key_padding_mask=temporal_padding,
        )[:, : self.num_classes]

        spatial_tokens = temporal_output.reshape(
            batch, self.num_patches, self.num_classes, self.dim
        ).permute(0, 2, 1, 3)
        spatial_tokens = spatial_tokens.reshape(
            batch * self.num_classes, self.num_patches, self.dim
        )
        spatial_tokens = spatial_tokens + self.spatial_position_embedding
        spatial_tokens = self.embedding_dropout(spatial_tokens)
        spatial_tokens = self.spatial_transformer(spatial_tokens)

        patch_logits = self.segmentation_head(spatial_tokens)
        patch_logits = patch_logits.reshape(
            batch,
            self.num_classes,
            self.patches_per_side,
            self.patches_per_side,
            self.patch_size,
            self.patch_size,
        )
        return patch_logits.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, self.num_classes, height, width
        )
