from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import scripts.train_e0 as train_e0
from scripts.train_e0 import load_formal_config


ROOT = Path(__file__).resolve().parents[1]
E4_PATH = ROOT / "configs" / "models" / "tsvit_e4_image_wavelet_no_doy_mexican_hat_k9.yaml"


def test_e4_formal_config_freezes_strict_image_only_k9_ablation() -> None:
    config = load_formal_config(E4_PATH)

    assert config["experiment"] == {
        "id": "E4-ImageWPE-K9",
        "name": "tsvit_image_nine_point_mexican_hat_k9",
        "seed": 42,
        "status": "not_run",
    }
    assert config["model"]["temporal_position_encoding"] == {"kind": "none"}
    assert config["model"]["wavelet"]["kernel_radius"] == 4
    assert train_e0.formal_output_for(config).name == (
        "e4_tsvit_image_nine_point_mexican_hat_k9_seed42"
    )


def test_e4_formal_config_rejects_k5_radius(tmp_path: Path) -> None:
    with E4_PATH.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["model"]["wavelet"]["kernel_radius"] = 2
    invalid_path = tmp_path / "bad-e4.yaml"
    invalid_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen design"):
        load_formal_config(invalid_path)
