"""Evaluate only a completed, Dev-selected, sealed checkpoint on the archived Native30m-v1 protocol."""

import time, hashlib
import numpy as np
import torch
from dgmm_moe.runtime import ROOT, lock, sha, path
from dgmm_moe.data import Assets, rows
from dgmm_moe.io_utils import save, write, state_sha
from dgmm_moe.training.common import expand


def test():
    from dgmm_moe.training.student import selected

    source = lock()
    m, st, cp = selected()
    from dgmm_moe.evaluation.native30m import (
        native_arrays,
        _deduplicate,
        score,
        EXPECTED_CDL_SHA256,
        EXPECTED_VALID_PIXELS,
    )
    import rasterio

    before = state_sha(m.state_dict())
    assets = Assets()
    rs = rows("test500.csv")
    assert len(rs) == 500
    cdlpath = path("test_cdl")
    assert sha(cdlpath) == EXPECTED_CDL_SHA256
    out = ROOT / "test"
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / "results.json").exists()
    ids = []
    targets = []
    preds = []
    h = hashlib.sha256()
    regions = {}
    start = time.time()
    with torch.no_grad(), rasterio.open(cdlpath) as cdl:
        for i, row in enumerate(rs):
            sample = assets.load(row, need_label=False)
            assert "target" not in sample
            bb = expand(sample)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(bb)
            p = o["logits"].float().softmax(1)[0, 1].cpu().numpy()
            a = native_arrays(p, tuple(sample["metadata"]["transform"]), cdl)
            v = a["valid"]
            ii = a["ids"][v]
            pp = a["prediction"][v]
            yy = a["target"][v]
            ids.append(ii)
            preds.append(pp)
            targets.append(yy)
            h.update(ii.tobytes())
            h.update(yy.tobytes())
            regions.setdefault(row["region_id"], []).append((ii, pp, yy))
            save(out / (row["sample_id"] + ".pt"), dict(ids=ii, prediction=pp, target=yy))
            if (i + 1) % 50 == 0:
                write(out / "progress.json", dict(tiles_completed=i + 1, total=500))
    p, y = _deduplicate(np.concatenate(ids), np.concatenate(preds), np.concatenate(targets))
    assert len(p) == EXPECTED_VALID_PIXELS
    assert h.hexdigest() == "046e8c97d7459135ddbf5193b5472e6ccbf106d33f5907af0f6f99e298c7a613"
    assert state_sha(m.state_dict()) == before
    overall = score(p, y)
    overall.update(
        corn_precision=overall["tp"] / max(1, overall["tp"] + overall["fp"]),
        corn_recall=overall["tp"] / max(1, overall["tp"] + overall["fn"]),
    )
    regional = {}
    for region, vals in regions.items():
        z, t = _deduplicate(*[np.concatenate([v[k] for v in vals]) for k in range(3)])
        regional[region] = score(z, t)
    write(
        out / "results.json",
        dict(
            status="completed",
            selected_epoch=st["epoch"],
            overall=overall,
            per_region=regional,
            tiles=500,
            checkpoint_sha256=sha(cp),
            reference_fingerprint=h.hexdigest(),
            test_pseudo_labels_loaded=False,
            source_lock=source,
            test_historically_reused=True,
            seconds=time.time() - start,
        ),
    )
