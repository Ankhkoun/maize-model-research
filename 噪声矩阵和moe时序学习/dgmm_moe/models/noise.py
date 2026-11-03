"""Train-only conditional transition experts and stable three-view likelihood."""

import torch
from torch import nn
from torch.nn import functional as F


class Noise(nn.Module):

    def __init__(self):
        super().__init__()
        k = 3
        alpha = torch.linspace(0.08, 0.12, k)
        beta = torch.linspace(0.12, 0.08, k)
        self.raw = nn.Parameter(torch.stack([alpha, beta, 1 - alpha - beta], 1).log())

    def matrices(self):
        rates = self.raw.softmax(-1)
        a, b = (rates[:, 0], rates[:, 1])
        return torch.stack([1 - a, a, b, 1 - b], -1).reshape(-1, 2, 2)

    def forward(self, logits, labels, weights, pixel_weight=None, alpha=1.0):
        assert labels.shape[1] == 3 and weights.shape[1:3] == (3, len(self.raw))
        t = torch.einsum("bvkxy,kco->bvcoxy", weights.to(self.raw.dtype), self.matrices()).reshape(
            logits.shape[0], 3, 2, 2, *logits.shape[-2:]
        )
        y = labels.clamp(0, 1).long()
        obs = t.gather(3, y[:, :, None, None].expand(-1, -1, 2, 1, -1, -1)).squeeze(3)
        log_evidence = (obs.clamp_min(1e-08).log() * (labels != 255)[:, :, None]).sum(1)
        log_joint = logits.to(self.raw.dtype).log_softmax(1) + log_evidence
        valid = labels[:, 0] != 255
        if not valid.any():
            raise RuntimeError("No valid center pseudo labels")
        nll = -log_joint.logsumexp(1)
        loss = nll[valid].mean()
        if pixel_weight is not None:
            assert not pixel_weight.requires_grad and pixel_weight.shape == valid.shape
            sample_weight = pixel_weight.detach().to(nll.dtype) * valid
            weighted = (nll * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
            loss = loss + alpha * (weighted - loss)
        posterior = log_joint.softmax(1)
        return (loss, posterior)


def conditions(source):
    """source[26,H,W] direct observation indicator; no labels or logits."""
    obs = F.avg_pool2d(source.float()[:, None], 4, 4)[:, 0]
    valid = obs > 0
    gap = torch.zeros_like(obs[0])
    longest = gap.clone()
    for slot in valid:
        gap = torch.where(slot, torch.zeros_like(gap), gap + 1)
        longest = torch.maximum(longest, gap)
    return torch.stack([obs.mean(0), longest / 26, *[x.mean(0) for x in obs.tensor_split(3, 0)]], 0)


def routing(s, mean, scale, centers):
    z = (s - mean[:, None, None]) / scale[:, None, None]
    d = (z[None] - centers[:, :, None, None]).square().sum(1)
    return (-d / float(len(mean))).softmax(0)
