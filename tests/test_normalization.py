import torch

from src.data.normalization import compute_band_stats


class _TinyDataset:
    def __init__(self) -> None:
        self.samples = [
            {
                "images": torch.tensor(
                    [
                        [[[1.0, 3.0]]], [[[10.0, 14.0]]],
                    ]
                ),
                "valid_mask": torch.tensor([True, False]),
                "split": "train",
            },
            {
                "images": torch.tensor(
                    [
                        [[[5.0, 7.0]]], [[[20.0, 22.0]]],
                    ]
                ),
                "valid_mask": torch.tensor([True, True]),
                "split": "train",
            },
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def test_compute_band_stats_uses_only_valid_train_frames() -> None:
    stats = compute_band_stats(_TinyDataset())
    expected = torch.tensor([1.0, 3.0, 5.0, 7.0, 20.0, 22.0])

    assert stats["tile_count"] == 2
    assert stats["count_per_band"] == [6]
    assert stats["mean"] == [expected.double().mean().item()]
    assert stats["std"] == [expected.double().std(correction=0).item()]


def test_compute_band_stats_rejects_non_train_sample() -> None:
    dataset = _TinyDataset()
    dataset.samples[1]["split"] = "validation"

    try:
        compute_band_stats(dataset)
    except ValueError as error:
        assert "Train-only" in str(error)
    else:
        raise AssertionError("expected Train-only validation error")
