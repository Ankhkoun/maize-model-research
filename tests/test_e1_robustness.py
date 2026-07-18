from pathlib import Path

import pytest
import torch
import yaml

from src.models.tsvit_segmentation import TSViTSegmentation


def _small_config(enabled: bool) -> dict:
    return {
        "image_size": 8,
        "patch_size": 2,
        "num_frames": 4,
        "num_channels": 3,
        "num_classes": 2,
        "dim": 8,
        "temporal_depth": 1,
        "spatial_depth": 1,
        "heads": 2,
        "dropout": 0.0,
        "emb_dropout": 0.0,
        "wavelet": {"enabled": enabled},
    }


def test_alpha_zero_preserves_common_parameter_gradients() -> None:
    torch.manual_seed(3)
    e0 = TSViTSegmentation(_small_config(enabled=False)).train()
    torch.manual_seed(4)
    e1 = TSViTSegmentation(_small_config(enabled=True)).train()
    e1.load_state_dict(e0.state_dict(), strict=False)
    assert e1.wavelet is not None
    with torch.no_grad():
        e1.wavelet.alpha.zero_()

    images = torch.randn(1, 4, 3, 8, 8)
    doy = torch.tensor([[95.0, 102.0, 109.0, 116.0]])
    valid_mask = torch.ones(1, 4, dtype=torch.bool)
    e0(images, doy, valid_mask).square().mean().backward()
    e1(images, doy, valid_mask).square().mean().backward()

    e1_parameters = dict(e1.named_parameters())
    for name, parameter in e0.named_parameters():
        assert parameter.grad is not None
        assert e1_parameters[name].grad is not None
        torch.testing.assert_close(
            e1_parameters[name].grad,
            parameter.grad,
            rtol=1.0e-6,
            atol=1.0e-8,
        )


def test_mixed_batch_with_all_invalid_sample_remains_finite() -> None:
    model = TSViTSegmentation(_small_config(enabled=True)).eval()
    images = torch.randn(2, 4, 3, 8, 8)
    doy = torch.tensor(
        [[95.0, 102.0, 109.0, 116.0], [float("nan")] * 4]
    )
    valid_mask = torch.tensor(
        [[True, True, True, True], [False, False, False, False]]
    )
    images[1] = float("nan")

    with torch.no_grad():
        logits = model(images, doy, valid_mask)

    assert logits.shape == (2, 2, 8, 8)
    assert torch.isfinite(logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_default_e1_config_cuda_forward_and_backward_are_finite() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / "models" / "tsvit_wpe_basic.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["model"]

    device = torch.device("cuda")
    model = TSViTSegmentation(config).to(device).train()
    image_size = config["image_size"]
    images = torch.randn(1, 26, 10, image_size, image_size, device=device)
    doy = torch.arange(94, 94 + 7 * 26, 7, device=device).float().unsqueeze(0)
    valid_mask = torch.ones(1, 26, dtype=torch.bool, device=device)
    target = torch.randint(0, 2, (1, image_size, image_size), device=device)

    logits = model(images, doy, valid_mask)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()

    assert logits.shape == (1, 2, image_size, image_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
