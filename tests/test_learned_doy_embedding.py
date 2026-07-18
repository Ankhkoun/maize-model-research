from pathlib import Path

import pytest
import torch
import yaml

from src.models.tsvit_segmentation import TSViTSegmentation


ROOT = Path(__file__).resolve().parents[1]


def _load_model_config(name: str) -> dict:
    path = ROOT / "configs" / "models" / name
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)["model"]


def _small_config() -> dict:
    config = _load_model_config("tsvit_baseline.yaml")
    config.update(
        {
            "image_size": 8,
            "num_frames": 3,
            "num_channels": 2,
            "dim": 16,
            "temporal_depth": 1,
            "spatial_depth": 1,
            "heads": 4,
        }
    )
    return config


def test_formal_e0_and_e1_configs_use_24px_windows_and_2px_patches() -> None:
    for name in ("tsvit_baseline.yaml", "tsvit_wpe_basic.yaml"):
        config = _load_model_config(name)
        assert config["image_size"] == 24
        assert config["patch_size"] == 2


def test_model_uses_calendar_doy_lookup_with_zero_padding() -> None:
    model = TSViTSegmentation(_small_config())

    assert isinstance(model.doy_embedding, torch.nn.Embedding)
    assert model.doy_embedding.num_embeddings == 367
    assert model.doy_embedding.padding_idx == 0
    torch.testing.assert_close(
        model.doy_embedding.weight[0],
        torch.zeros_like(model.doy_embedding.weight[0]),
    )


def test_valid_doy_must_be_integer_calendar_day() -> None:
    model = TSViTSegmentation(_small_config()).eval()
    images = torch.randn(1, 3, 2, 8, 8)
    valid_mask = torch.ones(1, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="integer calendar days"):
        model(images, torch.tensor([[95.0, 102.5, 109.0]]), valid_mask)


def test_invalid_padding_doy_is_safely_mapped_to_zero() -> None:
    model = TSViTSegmentation(_small_config()).eval()
    images = torch.randn(1, 3, 2, 8, 8)
    valid_mask = torch.tensor([[True, True, False]])
    doy = torch.tensor([[95.0, 102.0, float("nan")]])

    with torch.no_grad():
        logits = model(images, doy, valid_mask)

    assert torch.isfinite(logits).all()
