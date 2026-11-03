"""Shared Derived85 Dev evaluation, fixed threshold0.5; references used only for evaluation."""

import numpy as np
import torch
from dgmm_moe.data import batch


def expand(sample):
    return batch(
        {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
    )


@torch.no_grad()
def dev(m, assets, rs):
    from dgmm_moe.evaluation.metrics import score, metrics

    m.eval()
    counts = np.zeros(4, np.int64)
    tiles = []
    for row in rs:
        b = expand(assets.load(row, need_label=False))
        assert "target" not in b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(b)
        p = (
            o["logits"]
            .float()
            .softmax(1)[0, 1]
            .cpu()
            .numpy()[:255, :255]
            .reshape(85, 3, 85, 3)
            .mean((1, 3))
        )
        d = score(p >= 0.5, assets.dev_target(row["sample_id"]))
        c = [d[k] for k in ("tp", "fp", "fn", "tn")]
        counts += c
        tiles.append(dict(sample_id=row["sample_id"], counts=c))
    result = metrics(*map(int, counts))
    result["per_tile"] = tiles
    return result
