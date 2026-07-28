from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import scripts.train_e0 as train_e0
from scripts.train_e0 import load_formal_config


ROOT = Path(__file__).resolve().parents[1]
E5_PATH = ROOT / "configs" / "models" / "tsvit_e5_doy_wavelet_mexican_hat_k9.yaml"


def test_e5_formal_config_freezes_doy_plus_k9() -> None:
    config = load_formal_config(E5_PATH)

    assert config["experiment"]["id"] == "E5-DOY-K9WPE"
    assert config["model"]["temporal_position_encoding"] == {"kind": "doy_lookup"}
    assert config["model"]["wavelet"]["kernel_radius"] == 4
    assert train_e0.formal_output_for(config).name == (
        "e5_tsvit_doy_nine_point_mexican_hat_k9_seed42"
    )


def test_e5_formal_config_rejects_k5_or_missing_doy(tmp_path: Path) -> None:
    config = yaml.safe_load(E5_PATH.read_text(encoding="utf-8"))
    config["model"]["wavelet"]["kernel_radius"] = 2
    bad_k5 = tmp_path / "bad-k5.yaml"
    bad_k5.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen design"):
        load_formal_config(bad_k5)

    config["model"]["wavelet"]["kernel_radius"] = 4
    config["model"]["temporal_position_encoding"] = {"kind": "none"}
    bad_none = tmp_path / "bad-none.yaml"
    bad_none.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="calendar DOY"):
        load_formal_config(bad_none)
