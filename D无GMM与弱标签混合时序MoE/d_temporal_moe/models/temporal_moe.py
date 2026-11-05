"""Two proper generative experts: calendar and finite-shift marginalization."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class Evidence(nn.Module):

    def __init__(self, days, centers=None, scale=None, class_prior=None):
        super().__init__()
        days = torch.as_tensor(days, dtype=torch.float32)
        assert days.shape == (26,) and (days[1:] > days[:-1]).all()
        self.register_buffer("days", days)
        self.register_buffer("shifts", torch.tensor([-14.0, -7.0, 0.0, 7.0, 14.0]))
        self.register_buffer("shift_prior", torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]) / 9)
        self.register_buffer("expert_prior", torch.tensor([0.5, 0.5]))
        self.register_buffer(
            "class_prior",
            torch.as_tensor(
                class_prior if class_prior is not None else [0.5, 0.5], dtype=torch.float32
            ),
        )
        centers = (
            torch.zeros(2, 3, 26, 10)
            if centers is None
            else torch.as_tensor(centers, dtype=torch.float32)
        )
        assert centers.shape == (2, 3, 26, 10)
        self.templates = nn.Parameter(centers[None].repeat(2, 1, 1, 1, 1))
        scale = (
            torch.full((10,), 0.5) if scale is None else torch.as_tensor(scale, dtype=torch.float32)
        )
        self.raw_scale = nn.Parameter(torch.log(torch.expm1((scale - 0.05).clamp_min(0.01))))

    def scales(self):
        return 0.05 + F.softplus(self.raw_scale)

    def shifted(self, template, delta):
        query = (self.days - delta).clamp(self.days[0], self.days[-1])
        hi = torch.searchsorted(self.days, query.contiguous()).clamp(1, 25)
        lo = hi - 1
        frac = (query - self.days[lo]) / (self.days[hi] - self.days[lo])
        basis = (
            F.one_hot(lo, 26).to(template.dtype) * (1 - frac)[:, None]
            + F.one_hot(hi, 26).to(template.dtype) * frac[:, None]
        )
        return torch.einsum("uv,crvb->crub", basis, template)

    def component_means(self):
        a = self.templates[0].reshape(6, 26, 10)
        b = torch.stack([self.shifted(self.templates[1], s) for s in self.shifts], 2).reshape(
            30, 26, 10
        )
        return torch.cat([a, b], 0)

    def component_logdensity(self, x, observed):
        assert x.ndim == 3 and x.shape[1:] == (26, 10) and (observed.shape == x.shape[:2])
        means = self.component_means().flatten(1)
        var = self.scales().square()
        mask = observed.to(x.dtype)[..., None]
        constant = (mask * (x.square() / var + torch.log(2 * math.pi * var))).sum((1, 2))
        xm = (mask * x / var).flatten(1)
        mv = (mask.expand_as(x) / var).flatten(1)
        return -0.5 * (constant[:, None] - 2 * xm @ means.T + mv @ means.square().T)

    def loglikelihood(self, x, observed):
        component = self.component_logdensity(x, observed)
        a = torch.logsumexp(component[:, :6].reshape(-1, 2, 3), 2) - math.log(3)
        log_shift = self.shift_prior.log().log_softmax(0)
        b = torch.logsumexp(component[:, 6:].reshape(-1, 2, 3, 5) + log_shift, (2, 3)) - math.log(3)
        return torch.stack([a, b], 1)

    def fit_loss(self, x, observed, anchor):
        logl = self.loglikelihood(x, observed)
        conditional = torch.logsumexp(logl + self.expert_prior.log()[None, :, None], 1)
        return -(anchor.detach() * conditional).sum(1).mean()

    def forward(self, x, observed):
        logl = self.loglikelihood(x, observed)
        joint = (
            logl + self.class_prior.log()[None, None, :] + self.expert_prior.log()[None, :, None]
        )
        expert_prob = (logl + self.class_prior.log()[None, None, :]).softmax(2)
        weights = torch.logsumexp(joint, 2).softmax(1)
        probability = torch.logsumexp(joint, 1).softmax(1)
        return dict(
            probability=probability, weights=weights, expert_prob=expert_prob, loglikelihood=logl
        )


def soft_target(weak_label_weight, pseudo, probability, teacher=None, has_observation=None):
    """Mix weak labels with a detached probability; preserve teacher fallback."""
    weight = torch.as_tensor(weak_label_weight, dtype=probability.dtype, device=probability.device)
    if not torch.isfinite(weight).all() or not ((weight >= 0) & (weight <= 1)).all():
        raise ValueError("Weak-label weight must be finite and in [0, 1]")
    target = weight * (pseudo == 1).to(probability.dtype) + (1 - weight) * probability.detach()
    if teacher is not None and has_observation is not None:
        fallback = weight * (pseudo == 1).to(teacher.dtype) + (1 - weight) * teacher.detach()
        target = torch.where(has_observation, target, fallback)
    return target.detach()


def soft_ce(logits, positive, valid):
    assert not positive.requires_grad and positive.shape == valid.shape
    logp = logits.float().log_softmax(1)
    loss = -(positive * logp[:, 1] + (1 - positive) * logp[:, 0])
    return loss[valid].mean()
