"""Deterministic full-tile Validation evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.data.windows import LogitAccumulator, extract_windows, window_coordinates

from .metrics import ConfusionMatrix


def evaluate_tiles(
    model: nn.Module,
    dataset: Any,
    *,
    device: torch.device,
    window_size: int = 24,
    stride: int = 24,
    window_batch_size: int = 16,
    ignore_index: int = 255,
    amp: bool = True,
) -> dict[str, Any]:
    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be positive")
    overall = ConfusionMatrix(2, ignore_index)
    by_region: dict[str, ConfusionMatrix] = defaultdict(
        lambda: ConfusionMatrix(2, ignore_index)
    )
    loss_sum = 0.0
    supervised_pixels = 0
    was_training = model.training
    model.eval()
    evaluated_tiles = 0
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                if sample["split"] != "validation":
                    raise ValueError(
                        "Validation-only evaluator received a non-validation sample"
                    )
                images = torch.as_tensor(sample["images"])
                label = torch.as_tensor(sample["label"], dtype=torch.long)
                doy = torch.as_tensor(sample["doy"], dtype=torch.float32)
                valid_mask = torch.as_tensor(sample["valid_mask"], dtype=torch.bool)
                coordinates = window_coordinates(
                    label.shape[0], label.shape[1], window_size, stride
                )
                windows, _, kept = extract_windows(
                    images,
                    label,
                    coordinates,
                    window_size=window_size,
                    skip_all_ignore=False,
                    ignore_index=ignore_index,
                )
                accumulator = LogitAccumulator(2, label.shape[0], label.shape[1])
                for start in range(0, len(kept), window_batch_size):
                    stop = min(start + window_batch_size, len(kept))
                    batch = windows[start:stop].to(device=device, non_blocking=True)
                    batch_doy = doy.unsqueeze(0).expand(stop - start, -1).to(device)
                    batch_mask = valid_mask.unsqueeze(0).expand(stop - start, -1).to(device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=amp and device.type == "cuda",
                    ):
                        logits = model(batch, batch_doy, batch_mask)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(
                            f"non-finite validation logits for {sample['sample_id']}"
                        )
                    accumulator.add(logits, kept[start:stop])
                stitched = accumulator.finalize().float()
                prediction = stitched.argmax(dim=0)
                overall.update(prediction, label)
                by_region[sample["region_id"]].update(prediction, label)
                valid_pixels = int((label != ignore_index).sum().item())
                if valid_pixels > 0:
                    loss_sum += float(
                        F.cross_entropy(
                            stitched.unsqueeze(0),
                            label.unsqueeze(0),
                            ignore_index=ignore_index,
                            reduction="sum",
                        ).item()
                    )
                    supervised_pixels += valid_pixels
                evaluated_tiles += 1
    finally:
        model.train(was_training)

    if evaluated_tiles == 0 or supervised_pixels == 0:
        raise ValueError("Validation evaluation requires supervised tiles")
    metrics = overall.compute()
    metrics.update(
        {
            "loss": loss_sum / supervised_pixels,
            "supervised_pixels": supervised_pixels,
            "evaluated_tiles": evaluated_tiles,
            "per_region": {
                region_id: matrix.compute()
                for region_id, matrix in sorted(by_region.items())
            },
        }
    )
    return metrics
