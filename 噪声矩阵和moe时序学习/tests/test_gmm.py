import torch
import numpy as np
from dgmm_moe.models.noise import Noise
from dgmm_moe.reliability.gmm import fit
from tests.test_noise import check


def check_weighting():
    check()
    torch.manual_seed(429)
    m = Noise().double()
    z = torch.randn(1, 2, 4, 5, dtype=torch.float64, requires_grad=True)
    y = torch.randint(0, 2, (1, 3, 4, 5))
    y[:, 0, 0, 0] = 255
    y[:, 2, 1, 1] = 255
    w = torch.rand(1, 3, 3, 4, 5, dtype=torch.float64)
    w = w / w.sum(2, keepdim=True)
    base, gamma = m(z, y, w)
    ones = torch.ones(1, 4, 5, dtype=torch.float64)
    same, _ = m(z, y, w, ones, 1)
    assert torch.allclose(base, same, atol=1e-12)
    assert torch.allclose(
        torch.autograd.grad(base, z, retain_graph=True)[0],
        torch.autograd.grad(same, z, retain_graph=True)[0],
        atol=1e-12,
    )
    pixel = torch.linspace(0.05, 1, 20, dtype=torch.float64).reshape(1, 4, 5)
    valid = y[:, 0] != 255
    loss, post = m(z, y, w, pixel, 1)
    scaled, _ = m(z, y, w, pixel * 3, 1)
    assert torch.allclose(loss, scaled, atol=1e-12)
    expected = (z.softmax(1) - post) * (pixel * valid)[:, None] / (pixel * valid).sum()
    assert torch.allclose(torch.autograd.grad(loss, z, retain_graph=True)[0], expected, atol=1e-12)
    mixed, _ = m(z, y, w, pixel, 1 / 3)
    assert torch.allclose(mixed, base * 2 / 3 + loss / 3, atol=1e-12)
    assert fit(np.ones(64)) is None and fit(np.zeros(12)) is None
    rng = np.random.default_rng(42)
    values = np.r_[rng.normal(0.1, 0.01, 64), rng.normal(0.9, 0.01, 64)]
    q = fit(values)
    assert q is not None and q[:64].mean() > 0.99 and (q[64:].mean() < 0.01)
    print(
        "GMM weighted-joint CPU contracts passed: unit-weight forward/gradient parity, normalized p-posterior gradient, weight-scale invariance, ramp, ignored centers, EM and unavailable fallback"
    )


def test_contracts():
    check_weighting()
