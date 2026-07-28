from __future__ import annotations

import torch

from src.models.tsvit_segmentation import TSViTSegmentation


def _small_e3_model_config() -> dict:
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
        "temporal_position_encoding": {"kind": "none"},
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


def test_none_temporal_encoding_leaves_wavelet_logits_invariant_to_calendar_doy() -> None:
    """Breaks if the E3 temporal path injects calendar DOY into its tokens."""
    torch.manual_seed(23)
    model = TSViTSegmentation(_small_e3_model_config()).eval()
    images = torch.randn(1, 5, 10, 8, 8)
    valid_mask = torch.ones(1, 5, dtype=torch.bool)
    spring_doy = torch.tensor([[95.0, 102.0, 116.0, 130.0, 151.0]])
    autumn_doy = torch.tensor([[255.0, 262.0, 276.0, 290.0, 311.0]])

    with torch.no_grad():
        spring_logits = model(images, spring_doy, valid_mask)
        autumn_logits = model(images, autumn_doy, valid_mask)

    torch.testing.assert_close(spring_logits, autumn_logits, rtol=0.0, atol=0.0)

from pathlib import Path

import pytest
import yaml

import scripts.train_e0 as train_e0
from scripts.evaluate_test import FROZEN_TEST_ASSETS
from scripts.train_e0 import load_formal_config


ROOT = Path(__file__).resolve().parents[1]
E3_PATH = ROOT / "configs" / "models" / "tsvit_e3_image_wavelet_no_doy_mexican_hat_k5.yaml"


def test_e3_formal_config_freezes_the_no_doy_image_wavelet_ablation() -> None:
    config = load_formal_config(E3_PATH)

    assert config["experiment"] == {
        "id": "E3-ImageWPE",
        "name": "tsvit_image_five_point_mexican_hat_k5",
        "seed": 42,
        "status": "not_run",
    }
    assert config["model"]["temporal_position_encoding"] == {"kind": "none"}
    assert train_e0.formal_output_for(config).name == (
        "e3_tsvit_image_five_point_mexican_hat_k5_seed42"
    )


def test_e3_formal_config_rejects_doy_lookup(tmp_path: Path) -> None:
    with E3_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["model"]["temporal_position_encoding"] = {"kind": "doy_lookup"}
    invalid_path = tmp_path / "bad-e3.yaml"
    invalid_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="no temporal position encoding"):
        load_formal_config(invalid_path)

def test_e3_formal_entry_is_frozen_with_exact_test_asset() -> None:
    config = load_formal_config(E3_PATH)

    asset = FROZEN_TEST_ASSETS["E3-ImageWPE"]
    assert asset.experiment_id == "E3-ImageWPE"
    assert asset.checkpoint.as_posix().endswith(
        "e3_tsvit_image_five_point_mexican_hat_k5_seed42_rerun1/best.pt"
    )
    assert asset.sha256 == "83F9128719C60C87CA07A044B22DCA3FF4264E23316CD0CA8D088BF7FDC9FB85"
    assert asset.epoch == 15
    assert asset.validation_maize_iou == 0.9384604671989814
    assert config["experiment"]["id"] == asset.experiment_id