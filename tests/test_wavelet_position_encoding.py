import pytest
import torch

from src.models.wavelet_position_encoding import LearnableWaveletPositionEncoding


def _make_module(dim: int = 8) -> LearnableWaveletPositionEncoding:
    torch.manual_seed(0)
    return LearnableWaveletPositionEncoding(
        dim=dim,
        scale_init_days=(7.0, 17.5, 35.0),
        scale_min_days=3.5,
        scale_max_days=35.0,
        shift_init_days=(0.0, 0.0, 0.0),
        shift_max_abs_days=7.0,
        support_radius_days=42.0,
        alpha_init=0.01,
    )


def test_preserves_shape_and_exposes_three_bounded_bases() -> None:
    module = _make_module()
    tokens = torch.randn(2, 3, 5, 8)
    doy = torch.tensor([[1.0, 8.0, 15.0, 22.0, 29.0]]).expand(2, -1)
    valid_mask = torch.ones(2, 5, dtype=torch.bool)

    output = module(tokens, doy, valid_mask)

    assert output.shape == tokens.shape
    assert module.scales.shape == (3,)
    assert module.shifts.shape == (3,)
    assert torch.all(module.scales >= 3.5)
    assert torch.all(module.scales <= 35.0)
    assert torch.all(module.shifts >= -7.0)
    assert torch.all(module.shifts <= 7.0)


def test_parameterization_keeps_extreme_raw_values_inside_bounds() -> None:
    module = _make_module()

    with torch.no_grad():
        module.raw_scales.copy_(torch.tensor([-1.0e6, 0.0, 1.0e6]))
        module.raw_shifts.copy_(torch.tensor([-1.0e6, 0.0, 1.0e6]))

    assert torch.all(module.scales >= 3.5)
    assert torch.all(module.scales <= 35.0)
    assert torch.all(module.shifts >= -7.0)
    assert torch.all(module.shifts <= 7.0)


def test_padding_values_do_not_change_valid_query_outputs() -> None:
    module = _make_module()
    tokens = torch.randn(1, 2, 5, 8)
    changed = tokens.clone()
    changed[:, :, 3:] = 10000.0
    doy = torch.tensor([[1.0, 8.0, 15.0, 22.0, 29.0]])
    valid_mask = torch.tensor([[True, True, True, False, False]])

    original_output = module(tokens, doy, valid_mask)
    changed_output = module(changed, doy, valid_mask)

    torch.testing.assert_close(
        original_output[:, :, :3],
        changed_output[:, :, :3],
        rtol=0.0,
        atol=0.0,
    )


def test_all_invalid_sequence_has_zero_wavelet_residual() -> None:
    module = _make_module()
    tokens = torch.randn(2, 2, 4, 8)
    doy = torch.full((2, 4), float("nan"))
    valid_mask = torch.zeros(2, 4, dtype=torch.bool)

    output = module(tokens, doy, valid_mask)

    torch.testing.assert_close(output, tokens, rtol=0.0, atol=0.0)
    assert torch.isfinite(output).all()


def test_forward_and_backward_are_finite() -> None:
    module = _make_module()
    tokens = torch.randn(2, 3, 5, 8, requires_grad=True)
    doy = torch.tensor(
        [[95.0, 102.0, 116.0, 130.0, 151.0], [96.0, 104.0, 117.0, 131.0, 152.0]]
    )
    valid_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )

    output = module(tokens, doy, valid_mask)
    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("tokens", "doy", "valid_mask", "message"),
    [
        (torch.randn(1, 4, 8), torch.randn(1, 4), torch.ones(1, 4, dtype=torch.bool), "tokens"),
        (torch.randn(1, 2, 4, 8), torch.randn(1, 5), torch.ones(1, 4, dtype=torch.bool), "doy"),
        (torch.randn(1, 2, 4, 8), torch.randn(1, 4), torch.ones(1, 4), "valid_mask"),
    ],
)
def test_rejects_invalid_tensor_contracts(
    tokens: torch.Tensor,
    doy: torch.Tensor,
    valid_mask: torch.Tensor,
    message: str,
) -> None:
    module = _make_module()

    with pytest.raises(ValueError, match=message):
        module(tokens, doy, valid_mask)


def test_rejects_non_finite_doy_at_valid_frames() -> None:
    module = _make_module()
    tokens = torch.randn(1, 2, 4, 8)
    doy = torch.tensor([[1.0, 8.0, float("nan"), 22.0]])
    valid_mask = torch.ones(1, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="finite"):
        module(tokens, doy, valid_mask)
