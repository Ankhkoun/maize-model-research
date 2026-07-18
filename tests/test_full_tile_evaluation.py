import torch
from torch import nn

from src.training.evaluate import evaluate_tiles


class _PixelThresholdModel(nn.Module):
    def forward(
        self,
        images: torch.Tensor,
        doy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        foreground = images[:, 0, 0]
        return torch.stack((1.0 - foreground, foreground), dim=1)


class _TileDataset:
    def __init__(self, split: str = "validation") -> None:
        label = torch.zeros(6, 6, dtype=torch.long)
        label[2:, 2:] = 1
        images = label.float().reshape(1, 1, 6, 6)
        self.sample = {
            "images": images,
            "doy": torch.tensor([100.0]),
            "valid_mask": torch.tensor([True]),
            "label": label,
            "sample_id": "sample_a",
            "region_id": "region_a",
            "split": split,
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        assert index == 0
        return self.sample


def test_full_tile_evaluation_stitches_validation_logits() -> None:
    metrics = evaluate_tiles(
        _PixelThresholdModel(),
        _TileDataset(),
        device=torch.device("cpu"),
        window_size=4,
        stride=2,
        window_batch_size=2,
        amp=False,
    )

    assert metrics["maize_iou"] == 1.0
    assert metrics["maize_f1"] == 1.0
    assert metrics["confusion_matrix"] == [[20, 0], [0, 16]]
    assert metrics["per_region"]["region_a"]["maize_iou"] == 1.0
    assert metrics["evaluated_tiles"] == 1


def test_training_evaluator_rejects_test_split() -> None:
    try:
        evaluate_tiles(
            _PixelThresholdModel(),
            _TileDataset(split="test"),
            device=torch.device("cpu"),
            window_size=4,
            stride=2,
            window_batch_size=2,
            amp=False,
        )
    except ValueError as error:
        assert "Validation-only" in str(error)
    else:
        raise AssertionError("expected Validation-only guard")
