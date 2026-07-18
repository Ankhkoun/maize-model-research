"""Streaming Train-only band normalization statistics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def compute_band_stats(
    dataset: Any,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[float] | list[int] | int]:
    sums: torch.Tensor | None = None
    squared_sums: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    total = len(dataset)

    for index in range(total):
        sample = dataset[index]
        if sample["split"] != "train":
            raise ValueError("Train-only normalization received a non-train sample")
        images = torch.as_tensor(sample["images"])
        valid_mask = torch.as_tensor(sample["valid_mask"], dtype=torch.bool)
        if images.ndim != 4 or valid_mask.shape != (images.shape[0],):
            raise ValueError("normalization expects images [T,C,H,W] and valid_mask [T]")
        values = images[valid_mask].permute(1, 0, 2, 3).reshape(images.shape[1], -1)
        values = values.double()
        if values.numel() == 0 or not torch.isfinite(values).all():
            raise ValueError("normalization encountered no finite valid pixels")
        if sums is None:
            sums = torch.zeros(images.shape[1], dtype=torch.float64)
            squared_sums = torch.zeros_like(sums)
            counts = torch.zeros(images.shape[1], dtype=torch.int64)
        sums += values.sum(dim=1)
        squared_sums += values.square().sum(dim=1)
        counts += values.shape[1]
        if progress is not None:
            progress(index + 1, total)

    if sums is None or squared_sums is None or counts is None or (counts <= 0).any():
        raise ValueError("normalization dataset is empty")
    mean = sums / counts.double()
    variance = squared_sums / counts.double() - mean.square()
    std = variance.clamp_min(0.0).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("normalization produced invalid or zero standard deviations")
    return {
        "tile_count": total,
        "count_per_band": counts.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
