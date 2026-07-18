from copy import deepcopy

import pytest
import torch

from src.models.tsvit_segmentation import TSViTSegmentation


def _model_config(wavelet_enabled: bool) -> dict:
    return {
        "image_size": 8,
        "patch_size": 2,
        "num_frames": 5,
        "num_channels": 3,
        "num_classes": 2,
        "dim": 16,
        "temporal_depth": 1,
        "spatial_depth": 1,
        "heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        "wavelet": {
            "enabled": wavelet_enabled,
            "scale_init_days": [7.0, 17.5, 35.0],
            "scale_min_days": 3.5,
            "scale_max_days": 35.0,
            "shift_init_days": [0.0, 0.0, 0.0],
            "shift_max_abs_days": 7.0,
            "support_radius_days": 42.0,
            "alpha_init": 0.01,
            "eps": 1.0e-6,
        },
    }


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    images = torch.randn(2, 5, 3, 8, 8)
    doy = torch.tensor(
        [[95.0, 102.0, 116.0, 130.0, 151.0], [96.0, 104.0, 117.0, 131.0, 152.0]]
    )
    valid_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    return images, doy, valid_mask


def test_e1_produces_segmentation_logits_at_image_resolution() -> None:
    model = TSViTSegmentation(_model_config(wavelet_enabled=True)).eval()
    images, doy, valid_mask = _inputs()

    with torch.no_grad():
        logits = model(images, doy, valid_mask)

    assert logits.shape == (2, 2, 8, 8)
    assert torch.isfinite(logits).all()


def test_alpha_zero_e1_is_exactly_equivalent_to_e0() -> None:
    torch.manual_seed(10)
    e0 = TSViTSegmentation(_model_config(wavelet_enabled=False)).eval()
    torch.manual_seed(20)
    e1 = TSViTSegmentation(_model_config(wavelet_enabled=True)).eval()
    incompatible = e1.load_state_dict(e0.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("wavelet.") for key in incompatible.missing_keys)
    assert e1.wavelet is not None
    with torch.no_grad():
        e1.wavelet.alpha.zero_()

    images, doy, valid_mask = _inputs()
    with torch.no_grad():
        e0_logits = e0(images, doy, valid_mask)
        e1_logits = e1(images, doy, valid_mask)

    torch.testing.assert_close(e1_logits, e0_logits, rtol=0.0, atol=0.0)


def test_wavelet_receives_only_real_time_tokens_before_class_tokens() -> None:
    model = TSViTSegmentation(_model_config(wavelet_enabled=True)).eval()
    assert model.wavelet is not None
    observed_shapes: list[torch.Size] = []

    def record_shape(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        observed_shapes.append(args[0].shape)

    handle = model.wavelet.register_forward_pre_hook(record_shape)
    images, doy, valid_mask = _inputs()
    try:
        with torch.no_grad():
            model(images, doy, valid_mask)
    finally:
        handle.remove()

    assert observed_shapes == [torch.Size([2, 16, 5, 16])]


def test_padding_image_and_doy_values_do_not_change_logits() -> None:
    model = TSViTSegmentation(_model_config(wavelet_enabled=True)).eval()
    images, doy, valid_mask = _inputs()
    changed_images = images.clone()
    changed_doy = doy.clone()
    changed_images[1, 4] = 10000.0
    changed_doy[1, 4] = float("nan")

    with torch.no_grad():
        original = model(images, doy, valid_mask)
        changed = model(changed_images, changed_doy, valid_mask)

    torch.testing.assert_close(original, changed, rtol=0.0, atol=0.0)


def test_small_e1_forward_loss_and_backward_are_finite() -> None:
    model = TSViTSegmentation(_model_config(wavelet_enabled=True)).train()
    images, doy, valid_mask = _inputs()
    target = torch.randint(0, 2, (2, 8, 8))

    logits = model(images, doy, valid_mask)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda images, doy, mask: (images[:, :, :, :-1], doy, mask), "image"),
        (lambda images, doy, mask: (images[:, :-1], doy, mask), "time"),
        (lambda images, doy, mask: (images, doy[:, :-1], mask), "doy"),
        (lambda images, doy, mask: (images, doy, mask.float()), "valid_mask"),
    ],
)
def test_rejects_invalid_model_inputs(mutator, message: str) -> None:
    model = TSViTSegmentation(_model_config(wavelet_enabled=True))
    images, doy, valid_mask = _inputs()
    bad_images, bad_doy, bad_mask = mutator(images, doy, valid_mask)

    with pytest.raises(ValueError, match=message):
        model(bad_images, bad_doy, bad_mask)


def test_e0_and_e1_configs_differ_only_in_wavelet_settings() -> None:
    e0 = _model_config(wavelet_enabled=False)
    e1 = _model_config(wavelet_enabled=True)
    e0_without_wavelet = deepcopy(e0)
    e1_without_wavelet = deepcopy(e1)
    e0_without_wavelet.pop("wavelet")
    e1_without_wavelet.pop("wavelet")

    assert e0_without_wavelet == e1_without_wavelet
