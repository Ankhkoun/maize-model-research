from copy import deepcopy
from pathlib import Path

import torch
import yaml

from src.models.tsvit_segmentation import TSViTSegmentation


ROOT = Path(__file__).resolve().parents[1]
E0_PATH = ROOT / "configs" / "models" / "tsvit_baseline.yaml"
E1_PATH = ROOT / "configs" / "models" / "tsvit_wpe_basic.yaml"


def _load_document(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _load_model_config(path: Path) -> dict:
    return _load_document(path)["model"]


def test_e0_and_e1_yaml_share_all_non_wavelet_settings() -> None:
    e0 = _load_model_config(E0_PATH)
    e1 = _load_model_config(E1_PATH)
    e0_common = deepcopy(e0)
    e1_common = deepcopy(e1)
    e0_common.pop("wavelet")
    e1_common.pop("wavelet")

    assert e0_common == e1_common
    assert e0["wavelet"]["enabled"] is False
    assert e1["wavelet"]["enabled"] is True
    assert e1["wavelet"]["scale_init_days"] == [7.0, 17.5, 35.0]
    assert e1["wavelet"]["scale_min_days"] == 3.5
    assert e1["wavelet"]["scale_max_days"] == 35.0
    assert e1["wavelet"]["shift_max_abs_days"] == 7.0
    assert e1["wavelet"]["support_radius_days"] == 42.0


def test_yaml_configs_construct_models_and_support_small_smoke_forward() -> None:
    for path in (E0_PATH, E1_PATH):
        config = _load_model_config(path)
        config.update({"image_size": 8, "num_frames": 5})
        model = TSViTSegmentation(config).eval()
        images = torch.randn(1, 5, config["num_channels"], 8, 8)
        doy = torch.tensor([[95.0, 102.0, 116.0, 130.0, 151.0]])
        valid_mask = torch.ones(1, 5, dtype=torch.bool)

        with torch.no_grad():
            logits = model(images, doy, valid_mask)

        assert logits.shape == (1, config["num_classes"], 8, 8)
        assert torch.isfinite(logits).all()


def test_e0_and_e1_share_data_training_and_evaluation_policy() -> None:
    e0 = _load_document(E0_PATH)
    e1 = _load_document(E1_PATH)

    assert e0["data"] == e1["data"]
    assert e0["training"] == e1["training"]
    assert e0["evaluation"] == e1["evaluation"]
    assert e0["training"]["amp_backoff_factor"] == 0.5
    assert e0["training"]["amp_min_scale"] == 128.0
    assert e0["training"]["amp_max_backoffs_per_batch"] == 6
