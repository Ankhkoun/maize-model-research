import torch

from src.data.windows import (
    LogitAccumulator,
    augment_windows,
    extract_windows,
    window_coordinates,
    window_starts,
)


def test_window_starts_anchor_final_window() -> None:
    assert window_starts(256, 24, 24) == [
        0,
        24,
        48,
        72,
        96,
        120,
        144,
        168,
        192,
        216,
        232,
    ]
    assert len(window_coordinates(256, 256, 24, 24)) == 121


def test_extract_windows_skips_only_all_ignore_training_windows() -> None:
    images = torch.arange(2 * 1 * 8 * 8).reshape(2, 1, 8, 8).float()
    label = torch.full((8, 8), 255, dtype=torch.long)
    label[4:, 4:] = 1
    coordinates = [(0, 0), (4, 4)]

    windows, labels, kept = extract_windows(
        images, label, coordinates, window_size=4, skip_all_ignore=True
    )

    assert windows.shape == (1, 2, 1, 4, 4)
    assert labels.shape == (1, 4, 4)
    assert kept == [(4, 4)]


def test_stitching_averages_overlaps_without_holes() -> None:
    accumulator = LogitAccumulator(num_classes=1, height=6, width=6)
    accumulator.add(torch.full((1, 1, 4, 4), 1.0), [(0, 0)])
    accumulator.add(torch.full((1, 1, 4, 4), 3.0), [(0, 2)])
    accumulator.add(torch.full((1, 1, 4, 4), 5.0), [(2, 0)])
    accumulator.add(torch.full((1, 1, 4, 4), 7.0), [(2, 2)])

    output = accumulator.finalize()

    assert output[0, 0, 0].item() == 1.0
    assert output[0, 5, 5].item() == 7.0
    assert output[0, 2, 2].item() == 4.0


def test_spatial_augmentation_is_deterministic_and_keeps_alignment() -> None:
    images = torch.arange(2 * 1 * 1 * 3 * 3).reshape(2, 1, 1, 3, 3).float()
    labels = images[:, 0, 0].long()
    generator_a = torch.Generator().manual_seed(42)
    generator_b = torch.Generator().manual_seed(42)

    augmented_images_a, augmented_labels_a = augment_windows(
        images, labels, generator=generator_a
    )
    augmented_images_b, augmented_labels_b = augment_windows(
        images, labels, generator=generator_b
    )

    torch.testing.assert_close(augmented_images_a, augmented_images_b)
    torch.testing.assert_close(augmented_labels_a, augmented_labels_b)
    torch.testing.assert_close(augmented_images_a[:, 0, 0], augmented_labels_a.float())
