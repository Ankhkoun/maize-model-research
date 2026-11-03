"""Loss-only statistical target; no neural reliability prior or expert module."""

import numpy as np
import torch
from dgmm_moe.reliability.losses import raw_ce, patch_mean, eligible


def fit(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 32 or not np.isfinite(x).all() or x.std() < 1e-07:
        return None
    lo, hi = np.quantile(x, [0.05, 0.95])
    x = np.clip((x - lo) / max(hi - lo, 1e-08), 0, 1)
    means = np.quantile(x, [0.25, 0.75])
    var = np.full(2, max(x.var(), 0.001))
    pi = np.full(2, 0.5)
    for _ in range(40):
        a = np.log(pi + 1e-12) - 0.5 * (np.log(2 * np.pi * var) + (x[:, None] - means) ** 2 / var)
        a -= a.max(1, keepdims=True)
        q = np.exp(a)
        q /= q.sum(1, keepdims=True).clip(1e-12)
        mass = q.sum(0).clip(1e-06)
        pi = mass / len(x)
        means = (q * x[:, None]).sum(0) / mass
        var = ((q * (x[:, None] - means) ** 2).sum(0) / mass).clip(1e-05, 1)
    order = np.argsort(means)
    means, var, pi = (means[order], var[order], pi[order])
    if means[1] - means[0] < 0.08 or pi.min() < 0.03:
        return None
    a = np.log(pi + 1e-12) - 0.5 * (np.log(2 * np.pi * var) + (x[:, None] - means) ** 2 / var)
    a -= a.max(1, keepdims=True)
    q = np.exp(a)
    q /= q.sum(1, keepdims=True)
    return q[:, 0].astype(np.float32)


@torch.no_grad()
def targets(logits, b):
    valid, cls, _ = eligible(b)
    x = patch_mean(raw_ce(logits, b), b)
    s = torch.ones_like(x)
    available = torch.zeros_like(valid)
    modes = []
    for c in (0, 1):
        m = valid & (cls == c)
        values = x[m].cpu().numpy()
        q = fit(values)
        modes.append(
            dict(
                cls=c,
                count=len(values),
                mode=(
                    "gmm"
                    if q is not None
                    else "empty" if not len(values) else "unavailable_unit_weight"
                ),
            )
        )
        if q is not None:
            s[m] = torch.as_tensor(q, device=x.device)
            available[m] = True
    return (s.detach(), available, modes)
