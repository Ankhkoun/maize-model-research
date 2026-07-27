import torch

from src.models.wavelet_position_encoding import FivePointMexicanHatWaveletEncoding


def test_k9_radius_exposes_nine_symmetric_offsets():
    module = FivePointMexicanHatWaveletEncoding(dim=8, kernel_radius=4)

    torch.testing.assert_close(
        module.delta_t,
        torch.arange(-4, 5, dtype=torch.float32),
    )
    assert module.kernels.shape == (3, 9)
    torch.testing.assert_close(module.kernels.abs().sum(dim=-1), torch.ones(3))
