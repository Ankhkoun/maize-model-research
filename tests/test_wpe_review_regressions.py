import pytest
import torch

from src.models.tsvit_segmentation import TSViTSegmentation
from src.models.wavelet_position_encoding import LearnableWaveletPositionEncoding


def test_requires_exactly_three_wavelet_bases() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        LearnableWaveletPositionEncoding(
            dim=4,
            scale_init_days=(7.0, 17.5),
            shift_init_days=(0.0, 0.0),
        )


def test_scale_initialization_is_exact_and_has_gradient_at_upper_bound() -> None:
    module = LearnableWaveletPositionEncoding(dim=4)

    torch.testing.assert_close(
        module.scales,
        torch.tensor([7.0, 17.5, 35.0]),
        rtol=0.0,
        atol=0.0,
    )
    module.scales.sum().backward()

    assert module.raw_scales.grad is not None
    assert module.raw_scales.grad[-1] != 0


def test_large_finite_doy_differences_cannot_create_nan_before_masking() -> None:
    module = LearnableWaveletPositionEncoding(dim=4)
    tokens = torch.randn(1, 2, 3, 4)
    doy = torch.tensor([[-1.0e20, 1.0, 1.0e20]])
    valid_mask = torch.ones(1, 3, dtype=torch.bool)

    output = module(tokens, doy, valid_mask)

    assert torch.isfinite(output).all()


def test_model_accepts_float64_doy_with_float32_images() -> None:
    config = {
        "image_size": 8,
        "patch_size": 2,
        "num_frames": 3,
        "num_channels": 2,
        "num_classes": 2,
        "dim": 8,
        "temporal_depth": 1,
        "spatial_depth": 1,
        "heads": 2,
        "wavelet": {"enabled": True},
    }
    model = TSViTSegmentation(config).eval()
    images = torch.randn(1, 3, 2, 8, 8, dtype=torch.float32)
    doy = torch.tensor([[95.0, 102.0, 109.0]], dtype=torch.float64)
    valid_mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        logits = model(images, doy, valid_mask)

    assert logits.shape == (1, 2, 8, 8)
    assert torch.isfinite(logits).all()
