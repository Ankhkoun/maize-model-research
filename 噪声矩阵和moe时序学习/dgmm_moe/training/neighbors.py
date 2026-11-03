"""Train-only frozen F1 cosine neighbors; no NDRE, GT labels, or model probability scores in ranking."""

import json, time
import numpy as np
import torch
from torch.nn import functional as F
from dgmm_moe.runtime import ROOT, sha, plan, lock
from dgmm_moe.data import Assets, splits
from dgmm_moe.models.noise import conditions, routing
from dgmm_moe.io_utils import save, write, setup


@torch.no_grad()
def prepare():
    from dgmm_moe.training.teacher import initialized, expand

    setup(42)
    source = lock()
    out = ROOT / "prepared"
    out.mkdir(exist_ok=True)
    assert not (out / "complete.json").exists()
    assets = Assets()
    tr, _ = splits()
    m = initialized()
    m.eval()
    assert sha(__import__("contracts").PRETRAIN) == plan()["source_pretrained_sha256"]
    feats = []
    conds = []
    labels = []
    regions = []
    provenance = []
    masks = []
    start = time.time()
    gen = torch.Generator().manual_seed(42029)
    proj = torch.randn(1024, 64, generator=gen).cuda() / 8
    region_names = sorted({r["region_id"] for r in tr})
    assert len(region_names) >= 3
    for i, r in enumerate(tr):
        sample = assets.load(r)
        b = expand(sample)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = m.classifier.encode_features(b["agrifm_cube"])[0]
        assert f.shape == (1, 1024, 64, 64)
        f = F.normalize(f[0].float().permute(1, 2, 0).reshape(-1, 1024) @ proj, dim=1).cpu().half()
        sm = np.load(assets.cubes / r["sample_id"] / "source_map.npy", allow_pickle=False)
        masks.append(torch.from_numpy(sm[:, 2::4, 2::4] == 1).flatten(1).T)
        s = conditions(torch.from_numpy(sm == 1))
        y = sample["target"][2::4, 2::4]
        assert y.shape == (64, 64)
        feats.append(f)
        conds.append(s)
        labels.append(y)
        regions.append(region_names.index(r["region_id"]))
        provenance.append(
            dict(
                sample_id=r["sample_id"],
                label_sha256=sha(assets.labels / r["sample_id"] / "pseudo_label.npy"),
                source_map_sha256=sha(assets.cubes / r["sample_id"] / "source_map.npy"),
                cube_sha256=sha(assets.cubes / r["sample_id"] / "x_cube.npy"),
            )
        )
        if (i + 1) % 32 == 0:
            write(
                out / "progress.json",
                dict(stage="frozen_features", tiles=i + 1, total=1296, seconds=time.time() - start),
            )
    feats = torch.stack(feats)
    conds = torch.stack(conds)
    labels = torch.stack(labels)
    masks = torch.stack(masks)
    pick = torch.arange(4096).reshape(64, 64)[4::8, 4::8].reshape(-1)
    candidate = feats[:, pick].flatten(0, 1).float().cuda()
    cy = labels.flatten(1)[:, pick].flatten().cuda()
    cs = conds.flatten(2)[:, :, pick].permute(0, 2, 1).reshape(-1, 5)
    cr = torch.tensor(regions).repeat_interleave(64).cuda()
    cm = masks[:, pick].flatten(0, 1).cuda()
    valid = cy != 255
    candidate = candidate[valid]
    cy = cy[valid]
    cs = cs[valid.cpu()]
    cr = cr[valid]
    cm = cm[valid]
    assert all((int((cr == r).sum()) > 2 for r in range(len(region_names))))
    from sklearn.cluster import KMeans

    mean = cs.mean(0)
    scale = cs.std(0).clamp_min(0.05)
    centers = torch.from_numpy(
        KMeans(n_clusters=3, random_state=42, n_init=10)
        .fit(((cs - mean) / scale).numpy())
        .cluster_centers_
    ).float()
    cw = routing(cs.T[:, :, None], mean, scale, centers)[:, :, 0].T.cuda()
    save(
        out / "retrieval_model.pt",
        dict(
            projection=proj.cpu(),
            mean=mean,
            scale=scale,
            centers=centers,
            regions=region_names,
            candidate_count=len(cy),
        ),
    )
    for i, r in enumerate(tr):
        neighbors = []
        neighbor_weights = []
        queries = feats[i].float().cuda()
        for offset, q in enumerate(queries.split(256)):
            sim = q @ candidate.T
            shortlist = []
            for region in range(len(region_names)):
                if region == regions[i]:
                    continue
                ix = (cr == region).nonzero().flatten()
                shortlist.append(ix[sim[:, ix].topk(min(32, len(ix)), 1).indices])
            short = torch.cat(shortlist, 1)
            qm = masks[i, offset * 256 : offset * 256 + len(q)].cuda()
            common = qm[:, None] & cm[short]
            count = common.sum(-1)
            eligible = count >= 6
            for piece in common.tensor_split(3, -1):
                eligible &= piece.any(-1)
            distance = 1 - sim.gather(1, short)
            distance[~eligible] = torch.inf
            first = distance.argmin(1)
            j1 = short.gather(1, first[:, None])[:, 0]
            ok1 = distance.gather(1, first[:, None])[:, 0].isfinite()
            distance.masked_fill_(cr[short] == cr[j1, None], torch.inf)
            second = distance.argmin(1)
            j2 = short.gather(1, second[:, None])[:, 0]
            ok2 = distance.gather(1, second[:, None])[:, 0].isfinite()
            jj = torch.stack([j1, j2], 1)
            assert (
                (cr[j1] != regions[i]).all()
                and (cr[j2] != regions[i]).all()
                and (cr[j1[ok2]] != cr[j2[ok2]]).all()
            )
            yy = cy[jj].clone()
            yy[~torch.stack([ok1, ok2], 1)] = 255
            neighbors.append(yy.cpu())
            neighbor_weights.append(cw[jj].cpu())
        ny = torch.cat(neighbors).T.reshape(2, 64, 64)
        nw = torch.cat(neighbor_weights).permute(1, 2, 0).reshape(2, 3, 64, 64)
        own = routing(conds[i], mean, scale, centers)
        entry = dict(labels=ny, weights=torch.cat([own[None], nw]))
        save(out / "neighbors" / (r["sample_id"] + ".pt"), entry)
        if (i + 1) % 32 == 0:
            write(
                out / "progress.json",
                dict(stage="neighbors", tiles=i + 1, total=1296, seconds=time.time() - start),
            )
    write(out / "data_provenance.json", dict(labels=provenance, train_only=True))
    write(
        out / "complete.json",
        dict(
            source_lock=source,
            train_tiles=len(tr),
            candidate_count=len(cy),
            region_names=region_names,
            reference_labels_read=False,
            seconds=time.time() - start,
        ),
    )
    complete = json.loads((out / "complete.json").read_text())
    complete["neighbor_hashes"] = {
        r["sample_id"]: sha(out / "neighbors" / (r["sample_id"] + ".pt")) for r in tr
    }
    complete["data_provenance_sha256"] = sha(out / "data_provenance.json")
    write(out / "complete.json", complete)
