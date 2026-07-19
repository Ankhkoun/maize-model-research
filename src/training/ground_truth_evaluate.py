"""Full-tile Test evaluation against independent 30m reference labels."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.windows import LogitAccumulator, extract_windows, window_coordinates

from .metrics import ConfusionMatrix


IGNORE_INDEX = 255


def load_ground_truth_label(
    path: str | "Path", *, expected_shape: tuple[int, int] = (85, 85)
) -> torch.Tensor:
    """Load one independent reference tile with a strict binary/ignore contract."""
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing ground-truth label: {path}")
    label = np.load(path)
    if label.shape != expected_shape:
        raise ValueError(
            f"ground-truth label shape {label.shape} does not match {expected_shape}"
        )
    if not np.isin(label, (0, 1, IGNORE_INDEX)).all():
        raise ValueError("ground-truth label values must be 0, 1, or 255")
    return torch.from_numpy(np.asarray(label, dtype=np.int64))


def _crop_to_label_support(
    tensor: torch.Tensor, ground_truth_shape: tuple[int, int]
) -> torch.Tensor:
    height, width = (int(value) for value in ground_truth_shape)
    required_height, required_width = height * 3, width * 3
    actual_height, actual_width = tensor.shape[-2:]
    if actual_height - required_height not in {0, 1} or actual_width - required_width not in {
        0,
        1,
    }:
        raise ValueError(
            "model grid must equal the 3x3-expanded reference support, optionally "
            "with one discarded bottom/right border pixel"
        )
    return tensor[..., :required_height, :required_width]


def aggregate_native_30m_probabilities(
    probabilities: torch.Tensor, ground_truth_shape: tuple[int, int] = (85, 85)
) -> torch.Tensor:
    """Average class probabilities over exact 3x3 model-grid supports."""
    probabilities = torch.as_tensor(probabilities, dtype=torch.float32)
    if probabilities.ndim != 3 or probabilities.shape[0] != 2:
        raise ValueError("probabilities must have shape [2, height, width]")
    cropped = _crop_to_label_support(probabilities, ground_truth_shape)
    height, width = ground_truth_shape
    return cropped.reshape(2, height, 3, width, 3).mean(dim=(2, 4))


def upsample_30m_labels_to_model_grid(
    label: torch.Tensor, ground_truth_shape: tuple[int, int] = (85, 85)
) -> torch.Tensor:
    """Replicate each 30m reference cell to its exact 3x3 support."""
    label = torch.as_tensor(label, dtype=torch.long)
    if tuple(label.shape) != tuple(ground_truth_shape):
        raise ValueError(
            f"ground-truth label shape {tuple(label.shape)} does not match "
            f"{ground_truth_shape}"
        )
    return label.repeat_interleave(3, dim=0).repeat_interleave(3, dim=1)


def _new_state() -> dict[str, Any]:
    return {
        "overall": ConfusionMatrix(2, IGNORE_INDEX),
        "by_region": defaultdict(lambda: ConfusionMatrix(2, IGNORE_INDEX)),
        "loss_sum": 0.0,
        "supervised_pixels": 0,
        "evaluated_tiles": 0,
    }


def _update_state(
    state: dict[str, Any],
    probabilities: torch.Tensor,
    label: torch.Tensor,
    region_id: str,
) -> None:
    prediction = probabilities.argmax(dim=0)
    state["overall"].update(prediction, label)
    state["by_region"][region_id].update(prediction, label)
    valid_pixels = int((label != IGNORE_INDEX).sum().item())
    if valid_pixels:
        state["loss_sum"] += float(
            F.nll_loss(
                probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
                .log()
                .unsqueeze(0),
                label.unsqueeze(0),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            ).item()
        )
        state["supervised_pixels"] += valid_pixels
    state["evaluated_tiles"] += 1


def _finalize_state(state: dict[str, Any]) -> dict[str, Any]:
    if state["evaluated_tiles"] == 0 or state["supervised_pixels"] == 0:
        raise ValueError("ground-truth Test evaluation requires supervised tiles")
    metrics = state["overall"].compute()
    metrics.update(
        {
            "loss": state["loss_sum"] / state["supervised_pixels"],
            "supervised_pixels": state["supervised_pixels"],
            "evaluated_tiles": state["evaluated_tiles"],
            "per_region": {
                region_id: matrix.compute()
                for region_id, matrix in sorted(state["by_region"].items())
            },
        }
    )
    return metrics


def evaluate_ground_truth_tiles(
    model: nn.Module,
    dataset: Any,
    *,
    ground_truth_root: str | "Path",
    device: torch.device,
    window_size: int = 24,
    stride: int = 24,
    window_batch_size: int = 16,
    amp: bool = True,
    ground_truth_shape: tuple[int, int] = (85, 85),
) -> dict[str, dict[str, Any]]:
    """Stitch each Test tile once and derive native30m and upsampled10m metrics."""
    from pathlib import Path

    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be positive")
    ground_truth_root = Path(ground_truth_root)
    states = {"native30m": _new_state(), "upsampled10m": _new_state()}
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                if sample["split"] != "test":
                    raise ValueError("Test-only evaluator received a non-test sample")
                images = torch.as_tensor(sample["images"])
                source_label = torch.as_tensor(sample["label"], dtype=torch.long)
                doy = torch.as_tensor(sample["doy"], dtype=torch.float32)
                valid_mask = torch.as_tensor(sample["valid_mask"], dtype=torch.bool)
                coordinates = window_coordinates(
                    source_label.shape[0], source_label.shape[1], window_size, stride
                )
                windows, _, kept = extract_windows(
                    images,
                    source_label,
                    coordinates,
                    window_size=window_size,
                    skip_all_ignore=False,
                    ignore_index=IGNORE_INDEX,
                )
                accumulator = LogitAccumulator(2, source_label.shape[0], source_label.shape[1])
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
                            f"non-finite Test logits for {sample['sample_id']}"
                        )
                    accumulator.add(logits, kept[start:stop])
                probabilities = torch.softmax(accumulator.finalize().float(), dim=0)
                label30 = load_ground_truth_label(
                    ground_truth_root / sample["sample_id"] / "y_patch_30m.npy",
                    expected_shape=ground_truth_shape,
                )
                native_probabilities = aggregate_native_30m_probabilities(
                    probabilities, ground_truth_shape
                )
                probabilities10 = _crop_to_label_support(probabilities, ground_truth_shape)
                label10 = upsample_30m_labels_to_model_grid(
                    label30, ground_truth_shape
                )
                _update_state(
                    states["native30m"], native_probabilities, label30, sample["region_id"]
                )
                _update_state(
                    states["upsampled10m"], probabilities10, label10, sample["region_id"]
                )
    finally:
        model.train(was_training)
    return {name: _finalize_state(state) for name, state in states.items()}
