from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch

from scripts.train_e0 import build_optimizer, formal_output_for, load_formal_config
from src.training.schedule import WarmupCosineSchedule
from src.models.fixed_window_wpe import FixedWindowMexicanHatWaveletEncoding
from src.models.tsvit_segmentation import TSViTSegmentation


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/models/tsvit_e12_usa_fixed_window_wpe_seed42.yaml"
FIXED_CONFIG_PATH = ROOT / "configs/models/tsvit_e12_usa_fixed_window_wpe_fixedgamma02_seed42.yaml"
E13_CONFIG_PATH = ROOT / "configs/models/tsvit_e13_usa_cyclic_doy_fixed_window_wpe_fixedgamma02_seed42.yaml"
E16_CONFIG_PATH = ROOT / "configs/models/tsvit_e16_usa_cyclic_doy_fixed_window_wpe_k5_k9_k17_fixedgamma02_seed42.yaml"


def test_fixed_windows_and_kernels_match_preregistration() -> None:
    module = FixedWindowMexicanHatWaveletEncoding(dim=8)

    assert [offset.tolist() for offset in module.offsets] == [
        list(range(-2, 3)),
        list(range(-6, 7)),
        list(range(-12, 13)),
    ]
    assert module.sigmas.tolist() == [1.0, 3.0, 6.0]
    assert module.shifts.tolist() == [0.0, 0.0, 0.0]
    assert not any("sigma" in name or "shift" in name for name, _ in module.named_parameters())
    for kernel in module.kernels:
        assert bool((kernel > 0).any()) and bool((kernel < 0).any())
        assert float(kernel.sum()) == pytest.approx(0.0, abs=1.0e-6)
        assert float(kernel.abs().sum()) == pytest.approx(1.0, abs=1.0e-6)


def test_masked_boundary_and_all_invalid_are_finite() -> None:
    module = FixedWindowMexicanHatWaveletEncoding(dim=8)
    tokens = torch.randn(2, 3, 26, 8, requires_grad=True)
    valid = torch.ones(2, 26, dtype=torch.bool)
    valid[0, :24] = False
    valid[1] = False

    output = module(tokens, valid)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[1], tokens[1])
    output.square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()


def test_per_channel_fusion_and_serializable_diagnostics() -> None:
    module = FixedWindowMexicanHatWaveletEncoding(dim=8)
    weights = module.fusion_weights()
    assert weights.shape == (3, 8)
    torch.testing.assert_close(weights.sum(dim=0), torch.ones(8))

    tokens = torch.randn(2, 3, 26, 8)
    valid = torch.ones(2, 26, dtype=torch.bool)
    module(tokens, valid)
    diagnostics = module.diagnostics()
    json.dumps(diagnostics)
    assert diagnostics["windows"] == [5, 13, 25]
    assert diagnostics["radii"] == [2, 6, 12]
    assert diagnostics["sigmas"] == [1.0, 3.0, 6.0]
    assert diagnostics["shifts"] == [0.0, 0.0, 0.0]
    assert diagnostics["boundary_mode"] == "masked_renormalized"
    assert diagnostics["dtype"] == "float32"
    assert diagnostics["effective_residual_ratio"] >= 0.0
    assert 0.0 < diagnostics["gamma"] < 0.20
    assert set(diagnostics["fusion"]) == {"K5", "K13", "K25"}


def test_fixed_positive_gamma_is_not_trainable() -> None:
    module = FixedWindowMexicanHatWaveletEncoding(
        dim=8, gamma_init=0.02, gamma_parameterization="fixed_positive"
    )
    assert "raw_gamma" not in dict(module.named_parameters())
    assert "fixed_gamma" in dict(module.named_buffers())
    assert float(module.gamma_value) == pytest.approx(0.02)

    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1)
    tokens = torch.randn(2, 3, 26, 8)
    valid = torch.ones(2, 26, dtype=torch.bool)
    module(tokens, valid).square().mean().backward()
    optimizer.step()
    assert float(module.gamma_value) == pytest.approx(0.02)


def test_gamma_zero_strictly_matches_no_doy_baseline() -> None:
    config = load_formal_config(CONFIG_PATH)
    baseline_config = dict(config["model"])
    baseline_config["wavelet"] = {"enabled": False}
    baseline = TSViTSegmentation(baseline_config).eval()
    e12 = TSViTSegmentation(config["model"]).eval()
    incompatible = e12.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("wavelet.") for key in incompatible.missing_keys)
    assert e12.wavelet is not None
    with torch.no_grad():
        e12.wavelet.raw_gamma.fill_(-1.0e30)
    images = torch.randn(2, 26, 10, 24, 24)
    doy = torch.arange(1, 27, dtype=torch.float32).repeat(2, 1)
    valid = torch.ones(2, 26, dtype=torch.bool)
    torch.testing.assert_close(
        e12(images, doy, valid), baseline(images, doy, valid), rtol=0.0, atol=0.0
    )


def test_formal_e12_config_is_frozen() -> None:
    config = load_formal_config(CONFIG_PATH)
    assert config["experiment"] == {
        "id": "E12-USA-FixedWindow-WPE",
        "name": "usa_tsvit_nodoy_fixed_window_wpe_seed42",
        "seed": 42,
        "status": "authorized_train_validation",
    }
    assert config["model"]["temporal_position_encoding"] == {"kind": "none"}
    assert config["training"]["effective_batch_size"] == 32
    assert config["training"]["physical_batch_candidates"][0] == 32
    assert config["training"]["amp"] is False
    assert config["training"]["wavelet_lr_scale"] == pytest.approx(0.25)
    assert formal_output_for(config).name == "usa_e12_fixed_window_wpe_seed42"


def test_formal_fixed_gamma_e12_config_is_frozen() -> None:
    config = load_formal_config(FIXED_CONFIG_PATH)
    assert config["experiment"]["id"] == "E12-USA-FixedWindow-WPE-FixedGamma02"
    assert config["model"]["temporal_position_encoding"] == {"kind": "none"}
    assert config["model"]["wavelet"]["gamma_init"] == pytest.approx(0.02)
    assert config["model"]["wavelet"]["gamma_parameterization"] == "fixed_positive"
    assert formal_output_for(config).name == "usa_e12_fixed_window_wpe_fixedgamma02_seed42"


def test_cyclic_doy_is_separate_from_content_wpe() -> None:
    config = load_formal_config(E13_CONFIG_PATH)
    assert config["experiment"]["id"] == "E13-USA-CyclicDOY-FixedWindow-WPE-FixedGamma02"
    assert config["model"]["temporal_position_encoding"] == {
        "kind": "cyclic_sincos",
        "period": 365.0,
        "amplitude": 0.1,
    }
    model = TSViTSegmentation(config["model"]).eval()
    assert model.temporal_position_kind == "cyclic_sincos"
    assert model.wavelet is not None and model.wavelet.raw_gamma is None


def test_cyclic_doy_and_class_tokens_never_enter_fixed_window_wpe() -> None:
    config = load_formal_config(E13_CONFIG_PATH)
    model = TSViTSegmentation(config["model"]).eval()
    assert isinstance(model.wavelet, FixedWindowMexicanHatWaveletEncoding)
    captured: list[torch.Tensor] = []

    def capture(
        module: FixedWindowMexicanHatWaveletEncoding,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        del module, valid_mask
        captured.append(tokens.detach().clone())
        return tokens

    model.wavelet.forward = types.MethodType(capture, model.wavelet)
    images = torch.randn(1, 26, 10, 24, 24)
    valid = torch.ones(1, 26, dtype=torch.bool)
    doy_a = torch.arange(1, 27, dtype=torch.float32).unsqueeze(0)
    doy_b = doy_a + 100.0

    output_a = model(images, doy_a, valid)
    output_b = model(images, doy_b, valid)

    assert len(captured) == 2
    assert captured[0].shape == (1, model.num_patches, 26, model.dim)
    torch.testing.assert_close(captured[0], captured[1])
    assert not torch.equal(output_a, output_b)


def test_e16_fixed_windows_and_diagnostics_are_frozen() -> None:
    module = FixedWindowMexicanHatWaveletEncoding(
        dim=8,
        radii=(2, 4, 8),
        sigmas=(1.0, 2.0, 4.0),
        shifts=(0.0, 0.0, 0.0),
        gamma_init=0.02,
        gamma_parameterization="fixed_positive",
    )

    assert [offset.tolist() for offset in module.offsets] == [
        list(range(-2, 3)),
        list(range(-4, 5)),
        list(range(-8, 9)),
    ]
    assert module.sigmas.tolist() == [1.0, 2.0, 4.0]
    assert module.shifts.tolist() == [0.0, 0.0, 0.0]
    assert "raw_gamma" not in dict(module.named_parameters())
    assert float(module.gamma_value) == pytest.approx(0.02)
    assert not any("sigma" in name or "shift" in name for name, _ in module.named_parameters())
    for kernel in module.kernels:
        assert bool((kernel > 0).any()) and bool((kernel < 0).any())
        assert float(kernel.sum()) == pytest.approx(0.0, abs=1.0e-6)
        assert float(kernel.abs().sum()) == pytest.approx(1.0, abs=1.0e-6)

    tokens = torch.randn(2, 3, 26, 8)
    valid = torch.ones(2, 26, dtype=torch.bool)
    module(tokens, valid)
    diagnostics = module.diagnostics()
    json.dumps(diagnostics)
    assert diagnostics["windows"] == [5, 9, 17]
    assert diagnostics["radii"] == [2, 4, 8]
    assert diagnostics["sigmas"] == [1.0, 2.0, 4.0]
    assert diagnostics["shifts"] == [0.0, 0.0, 0.0]
    assert diagnostics["gamma"] == pytest.approx(0.02)
    assert set(diagnostics["fusion"]) == {"K5", "K9", "K17"}


def test_formal_e16_config_is_frozen_and_uses_cyclic_doy_content_only_wpe() -> None:
    config = load_formal_config(E16_CONFIG_PATH)
    assert config["experiment"] == {
        "id": "E16-USA-CyclicDOY-FixedWindow-WPE-K5-K9-K17-FixedGamma02",
        "name": "usa_tsvit_cyclic_doy_fixed_window_wpe_k5_k9_k17_fixedgamma02_seed42",
        "seed": 42,
        "status": "authorized_train_validation",
    }
    assert config["model"]["temporal_position_encoding"] == {
        "kind": "cyclic_sincos",
        "period": 365.0,
        "amplitude": 0.1,
    }
    assert config["model"]["wavelet"] == {
        "enabled": True,
        "kind": "fixed_window_mexican_hat",
        "radii": [2, 4, 8],
        "sigmas": [1.0, 2.0, 4.0],
        "shifts": [0.0, 0.0, 0.0],
        "gamma_init": 0.02,
        "gamma_parameterization": "fixed_positive",
        "gamma_max": 0.2,
        "eps": 1e-6,
        "boundary_mode": "masked_renormalized",
        "force_fp32": True,
    }
    assert config["training"]["effective_batch_size"] == 32
    assert config["training"]["amp"] is False
    assert formal_output_for(config).name == (
        "usa_e16_cyclic_doy_fixed_window_wpe_k5_k9_k17_fixedgamma02_seed42"
    )

    model = TSViTSegmentation(config["model"]).eval()
    assert model.temporal_position_kind == "cyclic_sincos"
    assert model.wavelet is not None and model.wavelet.raw_gamma is None
    assert model.wavelet.diagnostics()["windows"] == [5, 9, 17]


def test_wavelet_learning_rate_scale_survives_scheduler_updates() -> None:
    config = load_formal_config(CONFIG_PATH)
    model = TSViTSegmentation(config["model"])
    optimizer = build_optimizer(model, config["training"])
    scheduler = WarmupCosineSchedule(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        start_lr=config["training"]["start_lr"],
        base_lr=config["training"]["base_lr"],
        min_lr=config["training"]["min_lr"],
    )

    assert [group["group_name"] for group in optimizer.param_groups] == [
        "backbone",
        "wavelet",
    ]
    for _ in range(4):
        scheduler.step()
        backbone_lr, wavelet_lr = scheduler.get_last_lr()
        assert wavelet_lr == pytest.approx(0.25 * backbone_lr)
