import torch
from d_temporal_moe.models.noise import Noise, conditions, routing


def check():
    torch.manual_seed(42)
    k = 3
    m = Noise().double()
    z = torch.randn(1, 2, 3, 4, requires_grad=True, dtype=torch.float64)
    y = torch.randint(0, 2, (1, 3, 3, 4))
    w = torch.rand(1, 3, k, 3, 4)
    w = w / w.sum(2, keepdim=True)
    loss, gamma = m(z, y, w)
    grad = torch.autograd.grad(loss, z, retain_graph=True)[0]
    assert torch.allclose(grad, (z.softmax(1) - gamma) / 12, atol=1e-07)
    loss.backward()
    assert m.raw.grad.isfinite().all() and m.raw.grad.abs().sum() > 0
    t = m.matrices()
    assert torch.allclose(t.sum(-1), torch.ones(k, 2, dtype=t.dtype))
    assert (t[:, 0, 1] + t[:, 1, 0] < 1).all()
    y[:, 2] = 255
    loss, _ = m(z, y, w)
    mixed = torch.einsum("bvkxy,kco->bvcoxy", w.double(), m.matrices())
    probs = z.softmax(1)
    likelihood = torch.zeros_like(probs[:, 0])
    for c in (0, 1):
        val = probs[:, c]
        for v in (0, 1):
            val = val * mixed[:, v, c].gather(1, y[:, v, None]).squeeze(1)
        likelihood += val
    assert torch.allclose(loss, -likelihood.log().mean(), atol=1e-06)
    s = conditions(torch.ones(26, 256, 256))
    assert s.shape == (5, 64, 64) and s[1].max() == 0
    w = routing(s, torch.zeros(5), torch.ones(5), torch.zeros(3, 5))
    assert torch.allclose(w, torch.full_like(w, 1 / 3))
    print(
        "CPU contracts passed: gradients p-gamma, brute likelihood, 255 marginalization, stochastic positive-information matrices, routing"
    )


def test_contracts():
    check()
