import torch

from src.models.tsvit_segmentation import TSViTSegmentation


def test_k9_none_temporal_encoding_is_invariant_to_doy() -> None:
    torch.manual_seed(29)
    model = TSViTSegmentation(
        {
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
                "kernel_radius": 4,
            },
        }
    ).eval()
    images = torch.randn(1, 9, 10, 8, 8)
    valid_mask = torch.ones(1, 9, dtype=torch.bool)

    with torch.no_grad():
        spring_logits = model(images, torch.arange(90.0, 99.0).unsqueeze(0), valid_mask)
        autumn_logits = model(images, torch.arange(250.0, 259.0).unsqueeze(0), valid_mask)

    torch.testing.assert_close(spring_logits, autumn_logits, rtol=0.0, atol=0.0)
