"""Ordinary weak-label CE used for the shared CE3 initialization."""
from torch.nn import functional as F


def ordinary_ce(logits, batch):
    valid = batch["valid"]
    target = batch["target"].masked_fill(~valid, 255)
    losses = F.cross_entropy(logits.float(), target, ignore_index=255, reduction="none")
    return losses.sum() / valid.sum().clamp_min(1)
