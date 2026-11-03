"""Patch eligibility, unweighted classifier CE, and the CE3 objective."""

import torch
from torch.nn import functional as F


def eligible(b):
    y = b["target"]
    v = b["valid"]
    n = F.avg_pool2d(v.float()[:, None], 2, 2)[:, 0] * 4
    p = F.avg_pool2d((y == 1).float()[:, None], 2, 2)[:, 0] * 4 / n.clamp_min(1)
    return ((n >= 3) & ((p >= 0.75) | (p <= 0.25)) & b["observed20"].bool(), (p >= 0.5).long(), n)


def raw_ce(logits, b):
    return F.cross_entropy(
        logits.float(),
        b["target"].masked_fill(~b["valid"], 255),
        ignore_index=255,
        reduction="none",
    )


def patch_mean(x, b):
    n = F.avg_pool2d(b["valid"].float()[:, None], 2, 2)[:, 0] * 4
    return F.avg_pool2d((x * b["valid"])[:, None], 2, 2)[:, 0] * 4 / n.clamp_min(1)


def ordinary_ce(logits, b):
    """Unweighted weak-label CE; label255 is excluded."""
    return raw_ce(logits, b).sum() / b["valid"].sum().clamp_min(1)
