from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.training.ground_truth_evaluate import (
    aggregate_native_30m_probabilities,
    evaluate_ground_truth_tiles,
    load_ground_truth_label,
    upsample_30m_labels_to_model_grid,
)


def test_native_30m_aggregation_and_upsampling_use_exact_3x3_support() -> None:
    maize = torch.zeros(6, 6)
    maize[:3, 3:] = 1.0
    maize[3:, :3] = 1.0
    probabilities = torch.stack((1.0 - maize, maize))

    native = aggregate_native_30m_probabilities(probabilities, (2, 2))
    upsampled = upsample_30m_labels_to_model_grid(
        torch.tensor([[0, 1], [1, 255]]),
        (2, 2),
    )

    assert torch.equal(native.argmax(dim=0), torch.tensor([[0, 1], [1, 0]]))
    assert upsampled.shape == (6, 6)
    assert torch.equal(upsampled[:3, 3:], torch.ones(3, 3, dtype=torch.long))
    assert torch.equal(upsampled[3:, 3:], torch.full((3, 3), 255))


def test_ground_truth_loader_rejects_wrong_shape_and_value_domain(tmp_path: Path) -> None:
    label_path = tmp_path / "y_patch_30m.npy"
    np.save(label_path, np.zeros((2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        load_ground_truth_label(label_path, expected_shape=(2, 2))

    np.save(label_path, np.array([[0, 1], [2, 255]], dtype=np.uint8))
    with pytest.raises(ValueError, match="0, 1, or 255"):
        load_ground_truth_label(label_path, expected_shape=(2, 2))


class _PixelThresholdModel(nn.Module):
    def forward(
        self,
        images: torch.Tensor,
        doy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        maize = images[:, 0, 0]
        return torch.stack((1.0 - maize, maize), dim=1)


class _TestDataset:
    def __init__(self, split: str = "test") -> None:
        label30 = torch.tensor([[0, 1], [1, 0]])
        image = label30.repeat_interleave(3, 0).repeat_interleave(3, 1).float()
        self.sample = {
            "images": image.reshape(1, 1, 6, 6),
            "doy": torch.tensor([100.0]),
            "valid_mask": torch.tensor([True]),
            "label": torch.zeros(6, 6, dtype=torch.long),
            "sample_id": "sample_a",
            "region_id": "region_a",
            "split": split,
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        assert index == 0
        return self.sample


def test_evaluator_stitches_once_and_reports_both_ground_truth_scales(
    tmp_path: Path,
) -> None:
    label_root = tmp_path / "labels"
    sample_dir = label_root / "sample_a"
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "y_patch_30m.npy", np.array([[0, 1], [1, 0]], dtype=np.uint8))

    results = evaluate_ground_truth_tiles(
        _PixelThresholdModel(),
        _TestDataset(),
        ground_truth_root=label_root,
        device=torch.device("cpu"),
        window_size=4,
        stride=2,
        window_batch_size=2,
        amp=False,
        ground_truth_shape=(2, 2),
    )

    assert results["native30m"]["evaluated_tiles"] == 1
    assert results["native30m"]["supervised_pixels"] == 4
    assert results["native30m"]["confusion_matrix"] == [[2, 0], [0, 2]]
    assert results["upsampled10m"]["supervised_pixels"] == 36
    assert results["upsampled10m"]["confusion_matrix"] == [[18, 0], [0, 18]]


def test_evaluator_rejects_non_test_samples(tmp_path: Path) -> None:
    label_root = tmp_path / "labels"
    sample_dir = label_root / "sample_a"
    sample_dir.mkdir(parents=True)
    np.save(sample_dir / "y_patch_30m.npy", np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="Test-only"):
        evaluate_ground_truth_tiles(
            _PixelThresholdModel(),
            _TestDataset(split="validation"),
            ground_truth_root=label_root,
            device=torch.device("cpu"),
            window_size=4,
            stride=2,
            window_batch_size=2,
            amp=False,
            ground_truth_shape=(2, 2),
        )
