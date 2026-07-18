"""Deterministic spatial windows, augmentation, and logit stitching."""

from __future__ import annotations

from collections.abc import Sequence

import torch


Coordinate = tuple[int, int]


def window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= 0 or window_size <= 0 or stride <= 0:
        raise ValueError("length, window_size, and stride must be positive")
    if window_size > length:
        raise ValueError("window_size cannot exceed the image length")
    final = length - window_size
    starts = list(range(0, final + 1, stride))
    if starts[-1] != final:
        starts.append(final)
    return starts


def window_coordinates(
    height: int,
    width: int,
    window_size: int,
    stride: int,
) -> list[Coordinate]:
    rows = window_starts(height, window_size, stride)
    columns = window_starts(width, window_size, stride)
    return [(row, column) for row in rows for column in columns]


def extract_windows(
    images: torch.Tensor,
    label: torch.Tensor,
    coordinates: Sequence[Coordinate],
    *,
    window_size: int,
    skip_all_ignore: bool = False,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor, list[Coordinate]]:
    if images.ndim != 4 or label.ndim != 2:
        raise ValueError("images and label must have shapes [T,C,H,W] and [H,W]")
    if tuple(images.shape[-2:]) != tuple(label.shape):
        raise ValueError("images and label spatial shapes must match")
    image_windows: list[torch.Tensor] = []
    label_windows: list[torch.Tensor] = []
    kept: list[Coordinate] = []
    height, width = label.shape
    for row, column in coordinates:
        if row < 0 or column < 0 or row + window_size > height or column + window_size > width:
            raise ValueError(f"window coordinate is out of bounds: {(row, column)}")
        target = label[row : row + window_size, column : column + window_size]
        if skip_all_ignore and bool((target == ignore_index).all()):
            continue
        image_windows.append(
            images[:, :, row : row + window_size, column : column + window_size]
        )
        label_windows.append(target)
        kept.append((row, column))
    if not image_windows:
        return (
            images.new_empty((0, *images.shape[:2], window_size, window_size)),
            label.new_empty((0, window_size, window_size)),
            kept,
        )
    return torch.stack(image_windows), torch.stack(label_windows), kept


def augment_windows(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 5 or labels.ndim != 3 or images.shape[0] != labels.shape[0]:
        raise ValueError("augmentation expects images [B,T,C,H,W] and labels [B,H,W]")
    augmented_images: list[torch.Tensor] = []
    augmented_labels: list[torch.Tensor] = []
    for image, label in zip(images, labels, strict=True):
        rotations = int(torch.randint(0, 4, (1,), generator=generator).item())
        horizontal = bool(torch.randint(0, 2, (1,), generator=generator).item())
        vertical = bool(torch.randint(0, 2, (1,), generator=generator).item())
        image = torch.rot90(image, rotations, dims=(-2, -1))
        label = torch.rot90(label, rotations, dims=(-2, -1))
        if horizontal:
            image = image.flip(-1)
            label = label.flip(-1)
        if vertical:
            image = image.flip(-2)
            label = label.flip(-2)
        augmented_images.append(image)
        augmented_labels.append(label)
    return torch.stack(augmented_images), torch.stack(augmented_labels)


class LogitAccumulator:
    def __init__(
        self,
        num_classes: int,
        height: int,
        width: int,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if num_classes <= 0 or height <= 0 or width <= 0:
            raise ValueError("accumulator dimensions must be positive")
        self.sums = torch.zeros(num_classes, height, width, dtype=dtype)
        self.counts = torch.zeros(height, width, dtype=torch.int64)

    def add(self, logits: torch.Tensor, coordinates: Sequence[Coordinate]) -> None:
        if logits.ndim != 4 or logits.shape[0] != len(coordinates):
            raise ValueError("logits must be [B,K,H,W] and match coordinates")
        if logits.shape[1] != self.sums.shape[0] or logits.shape[2] != logits.shape[3]:
            raise ValueError("logit class/window shapes do not match the accumulator")
        size = logits.shape[-1]
        for logit, (row, column) in zip(logits, coordinates, strict=True):
            if row < 0 or column < 0 or row + size > self.sums.shape[1] or column + size > self.sums.shape[2]:
                raise ValueError(f"logit coordinate is out of bounds: {(row, column)}")
            self.sums[:, row : row + size, column : column + size] += logit.detach().to(
                dtype=self.sums.dtype, device=self.sums.device
            )
            self.counts[row : row + size, column : column + size] += 1

    def finalize(self) -> torch.Tensor:
        if bool((self.counts == 0).any()):
            raise ValueError("cannot finalize logits with uncovered pixels")
        return self.sums / self.counts.unsqueeze(0)
