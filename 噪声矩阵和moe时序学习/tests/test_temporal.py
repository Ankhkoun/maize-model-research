import math
import torch
from dgmm_moe.models.temporal_moe import Evidence, soft_target, soft_ce


def check():
    torch.manual_seed(30)
    days = torch.arange(26) * 7 + 94.0
    days[-1] += 0.5
    e = Evidence(days, torch.randn(2, 3, 26, 10) * 0.15, class_prior=[0.6, 0.4]).double()
    x = torch.randn(7, 26, 10, dtype=torch.float64)
    obs = torch.rand(7, 26) > 0.3
    obs[0] = False
    component = e.component_logdensity(x, obs)
    means = e.component_means()
    var = e.scales().square()
    direct = -0.5 * (
        ((x[:, None] - means[None]).square() / var + torch.log(2 * math.pi * var))
        * obs[:, None, :, None]
    ).sum((2, 3))
    assert torch.allclose(component, direct, atol=1e-09)
    corrupted = torch.where(obs[..., None], x, x + 1000)
    assert torch.allclose(e.component_logdensity(corrupted, obs), component, atol=1e-08)
    out = e(x, obs)
    assert torch.allclose(out["probability"].sum(1), torch.ones(7, dtype=torch.float64))
    assert torch.allclose(
        (out["weights"][:, :, None] * out["expert_prob"]).sum(1), out["probability"], atol=1e-12
    )
    assert torch.allclose(out["probability"][0], e.class_prior, atol=1e-07)
    assert torch.allclose(out["weights"][0], e.expert_prior, atol=1e-07)
    assert (
        e.loglikelihood(x, obs)[:, 1] <= component[:, 6:].reshape(-1, 2, 15).max(2).values + 1e-08
    ).all()
    with torch.no_grad():
        e.shift_prior.copy_(torch.tensor([0, 0, 1, 0, 0]))
    l = e.loglikelihood(x, obs)
    assert torch.allclose(l[:, 0], l[:, 1], atol=1e-09)
    with torch.no_grad():
        e.shift_prior.copy_(torch.tensor([1, 2, 3, 2, 1], dtype=torch.float64) / 9)
    anchor = torch.tensor([[0.4, 0.6]], dtype=torch.float64).expand(7, -1)
    loss = e.fit_loss(x, obs, anchor)
    loss.backward()
    assert all(
        (
            p.grad is not None and p.grad.isfinite().all() and (p.grad.abs().sum() > 0)
            for p in e.parameters()
        )
    )
    assert (e.scales() >= 0.05).all()
    q = torch.tensor([0.0, 0.2, 1.0])
    pseudo = torch.tensor([0, 0, 1])
    prob = torch.tensor([0.8, 0.8, 0.2], requires_grad=True)
    teacher = torch.tensor([0.3, 0.4, 0.7])
    t = soft_target(q, pseudo, prob, teacher, torch.tensor([True, True, False]))
    assert torch.allclose(t, torch.tensor([0.8, 0.64, 1.0])) and (not t.requires_grad)
    z = torch.randn(1, 2, 2, 2, requires_grad=True)
    positive = torch.tensor([[[0.1, 0.3], [0.7, 1.0]]])
    v = torch.tensor([[[True, True], [False, True]]])
    ce = soft_ce(z, positive, v)
    ce.backward()
    target = torch.stack([1 - positive, positive], 1)
    expected = (z.softmax(1) - target) * v[:, None] / v.sum()
    assert torch.allclose(z.grad, expected, atol=1e-07)
    print(
        "CPU evidence contracts passed: exact observed Gaussian, ignored-value invariance, true shift marginalization, posterior mixture identity, zero-observation priors, delta0 calendar parity, both parameter gradients, soft targets and p-target gradient"
    )


def test_contracts():
    check()
