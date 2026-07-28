from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

import scripts.train_e0 as train_e0
from scripts.evaluate_test import FROZEN_TEST_ASSETS
from scripts.train_e0 import load_formal_config
from src.models.tsvit_segmentation import TSViTSegmentation
from src.models.wavelet_position_encoding import FivePointMexicanHatWaveletEncoding


ROOT = Path(__file__).resolve().parents[1]
E0_PATH = ROOT / "configs" / "models" / "tsvit_baseline.yaml"
E2_WPEONLY_PATH = (
    ROOT / "configs" / "models" / "tsvit_e2_wpeonly_doy_mexican_hat_k5.yaml"
)


def _load_document(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _small_wpeonly_model_config() -> dict:
    return {
        "image_size": 8,
        "patch_size": 2,
        "num_frames": 5,
        "num_channels": 10,
        "num_classes": 2,
        "dim": 16,
        "temporal_depth": 1,
        "spatial_depth": 1,
        "heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        "temporal_position_encoding": {"kind": "doy_lookup"},
        "wavelet": {
            "enabled": True,
            "kind": "five_point_mexican_hat",
            "scales_init": [0.75, 1.0, 1.25],
            "scale_bounds": [0.5, 1.5],
            "shifts_init": [0.0, 0.0, 0.0],
            "shift_bounds": [-1.0, 1.0],
            "gamma_init": 0.001,
            "eps": 1.0e-6,
        },
    }


def test_e2_wpeonly_config_is_a_single_factor_e0_ablation() -> None:
    e0 = _load_document(E0_PATH)
    e2 = _load_document(E2_WPEONLY_PATH)

    assert e2["experiment"] == {
        "id": "E2-WPEonly",
        "name": "tsvit_doy_five_point_mexican_hat_k5",
        "seed": 42,
        "status": "not_run",
    }
    assert e2["model"]["temporal_position_encoding"] == {"kind": "doy_lookup"}
    assert e2["model"]["wavelet"] == {
        "enabled": True,
        "kind": "five_point_mexican_hat",
        "scales_init": [0.75, 1.0, 1.25],
        "scale_bounds": [0.5, 1.5],
        "shifts_init": [0.0, 0.0, 0.0],
        "shift_bounds": [-1.0, 1.0],
        "gamma_init": 0.001,
        "eps": 1.0e-6,
    }

    e0_common = deepcopy(e0)
    e2_common = deepcopy(e2)
    e0_common.pop("experiment")
    e2_common.pop("experiment")
    e0_common["model"].pop("wavelet")
    e2_common["model"].pop("wavelet")
    e2_common["model"].pop("temporal_position_encoding")
    assert e2_common == e0_common


def test_e2_wpeonly_formal_entry_is_frozen_with_exact_test_asset() -> None:
    config = load_formal_config(E2_WPEONLY_PATH)

    assert config["experiment"]["id"] == "E2-WPEonly"
    assert train_e0.formal_output_for(config).name == (
        "e2_wpeonly_tsvit_doy_mexican_hat_k5_seed42"
    )
    asset = FROZEN_TEST_ASSETS["E2-WPEonly"]
    assert asset.epoch == 11
    assert asset.validation_maize_iou == 0.9375021026721523
def test_e2_wpeonly_rejects_learned_temporal_slots(tmp_path: Path) -> None:
    config = _load_document(E2_WPEONLY_PATH)
    config["model"]["temporal_position_encoding"]["kind"] = "learned_slot"
    path = tmp_path / "bad-e2-wpeonly.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="calendar DOY lookup"):
        load_formal_config(path)


def test_e2_wpeonly_rejects_wavelet_parameter_drift(tmp_path: Path) -> None:
    config = _load_document(E2_WPEONLY_PATH)
    config["model"]["wavelet"]["gamma_init"] = 0.01
    path = tmp_path / "bad-e2-wpeonly.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen design"):
        load_formal_config(path)


def test_e2_wpeonly_routes_content_through_wpe_before_adding_doy() -> None:
    torch.manual_seed(7)
    config = _small_wpeonly_model_config()
    model = TSViTSegmentation(config).eval()
    assert model.temporal_position_kind == "doy_lookup"
    assert model.temporal_slot_embedding is None
    assert isinstance(model.wavelet, FivePointMexicanHatWaveletEncoding)

    captured: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {}

    def record_patch(_module, _inputs, output):
        captured["patch"] = output.detach().clone()

    def record_wavelet_input(_module, inputs):
        captured["wavelet_inputs"] = tuple(item.detach().clone() for item in inputs)

    def record_wavelet_output(_module, _inputs, output):
        captured["wavelet_output"] = output.detach().clone()

    def record_doy(_module, _inputs, output):
        captured["doy"] = output.detach().clone()

    class RecordingTemporal(nn.Module):
        def forward(self, tokens, *, key_padding_mask=None):
            del key_padding_mask
            captured["temporal_input"] = tokens.detach().clone()
            return tokens

    model.patch_embedding.register_forward_hook(record_patch)
    model.wavelet.register_forward_pre_hook(record_wavelet_input)
    model.wavelet.register_forward_hook(record_wavelet_output)
    model.doy_embedding.register_forward_hook(record_doy)
    model.temporal_transformer = RecordingTemporal()

    images = torch.randn(1, 5, 10, 8, 8)
    doy = torch.tensor([[95.0, 102.0, 116.0, 130.0, 151.0]])
    valid_mask = torch.tensor([[True, True, True, True, False]])

    with torch.no_grad():
        model(images, doy, valid_mask)

    patch = captured["patch"]
    assert isinstance(patch, torch.Tensor)
    expected_content = patch.reshape(1, 5, model.num_patches, model.dim).permute(
        0, 2, 1, 3
    )
    wavelet_inputs = captured["wavelet_inputs"]
    assert isinstance(wavelet_inputs, tuple)
    assert len(wavelet_inputs) == 2
    torch.testing.assert_close(wavelet_inputs[0], expected_content)
    torch.testing.assert_close(wavelet_inputs[1], valid_mask)

    wavelet_output = captured["wavelet_output"]
    doy_output = captured["doy"]
    temporal_input = captured["temporal_input"]
    assert isinstance(wavelet_output, torch.Tensor)
    assert isinstance(doy_output, torch.Tensor)
    assert isinstance(temporal_input, torch.Tensor)
    real_tokens = temporal_input[:, model.num_classes :].reshape(
        1, model.num_patches, 5, model.dim
    )
    torch.testing.assert_close(real_tokens, wavelet_output + doy_output[:, None])
    assert wavelet_inputs[0].shape[2] == 5
    assert temporal_input.shape[1] == model.num_classes + 5


def test_e2_wpeonly_invalid_keys_and_queries_are_isolated() -> None:
    torch.manual_seed(11)
    module = FivePointMexicanHatWaveletEncoding(dim=8)
    tokens = torch.randn(1, 2, 5, 8)
    changed = tokens.clone()
    changed[:, :, 3:] = 10000.0
    valid_mask = torch.tensor([[True, True, True, False, False]])

    output = module(tokens, valid_mask)
    changed_output = module(changed, valid_mask)

    torch.testing.assert_close(output[:, :, :3], changed_output[:, :, :3])
    torch.testing.assert_close(output[:, :, 3:], tokens[:, :, 3:])


def test_disabling_e2_wpeonly_is_exactly_e0_model_path() -> None:
    torch.manual_seed(13)
    e0_config = deepcopy(_small_wpeonly_model_config())
    e0_config.pop("temporal_position_encoding")
    e0_config["wavelet"] = {"enabled": False}
    disabled_config = deepcopy(_small_wpeonly_model_config())
    disabled_config["wavelet"] = {"enabled": False}

    e0 = TSViTSegmentation(e0_config).eval()
    disabled = TSViTSegmentation(disabled_config).eval()
    disabled.load_state_dict(e0.state_dict(), strict=True)
    images = torch.randn(2, 5, 10, 8, 8)
    doy = torch.tensor(
        [[95.0, 102.0, 116.0, 130.0, 151.0], [96.0, 103.0, 117.0, 131.0, 152.0]]
    )
    valid_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )

    with torch.no_grad():
        expected = e0(images, doy, valid_mask)
        actual = disabled(images, doy, valid_mask)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_e2_wpeonly_cuda_forward_backward_is_finite() -> None:
    torch.manual_seed(17)
    model = TSViTSegmentation(_small_wpeonly_model_config()).cuda().train()
    images = torch.randn(2, 5, 10, 8, 8, device="cuda", requires_grad=True)
    doy = torch.tensor(
        [[95.0, 102.0, 116.0, 130.0, 151.0], [96.0, 103.0, 117.0, 131.0, 152.0]],
        device="cuda",
    )
    valid_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]],
        device="cuda",
    )

    logits = model(images, doy, valid_mask)
    loss = logits.float().square().mean()
    loss.backward()

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert images.grad is not None and torch.isfinite(images.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
