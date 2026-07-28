import torch

from src.models.tsvit_segmentation import TSViTSegmentation


def _small_e5_model_config() -> dict:
    return {
        "image_size": 8,
        "patch_size": 2,
        "num_frames": 9,
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
            "kernel_radius": 4,
        },
    }

def test_e5_uses_calendar_doy_and_k9_content_wavelet() -> None:
    torch.manual_seed(31)
    model = TSViTSegmentation(_small_e5_model_config()).eval()
    images = torch.randn(1, 9, 10, 8, 8)
    valid_mask = torch.ones(1, 9, dtype=torch.bool)

    with torch.no_grad():
        spring = model(images, torch.arange(90.0, 99.0).unsqueeze(0), valid_mask)
        autumn = model(images, torch.arange(250.0, 259.0).unsqueeze(0), valid_mask)

    assert model.wavelet is not None
    assert model.wavelet.delta_t.tolist() == list(range(-4, 5))
    assert not torch.equal(spring, autumn)
