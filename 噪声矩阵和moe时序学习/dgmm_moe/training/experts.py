"""Frozen teacher -> observed-only temporal evidence fit -> dense fixed targets -> FPN-only continuation."""

import json, time, shutil
import numpy as np
import torch
from torch.nn import functional as F
from dgmm_moe.runtime import ROOT, plan, lock, sha, PRETRAIN
from dgmm_moe.io_utils import setup, save, write, state_sha
from dgmm_moe.data import Assets, splits
from dgmm_moe.training.initialization import student_initialized as initialized
from dgmm_moe.training.common import expand
from dgmm_moe.models.temporal_moe import Evidence, soft_target
from dgmm_moe.observations import read as temporal_read


def load_evidence():
    complete = json.loads((ROOT / "prepared/fit_complete.json").read_text())
    assert complete["source_lock"] == lock()
    f = ROOT / "prepared/evidence.pt"
    assert sha(f) == complete["evidence_sha256"]
    s = torch.load(f, map_location="cpu", weights_only=False)["evidence"]
    m = Evidence(s["days"])
    m.load_state_dict(s, strict=True)
    m.cuda().eval()
    m.requires_grad_(False)
    return m


@torch.no_grad()
def prepare():
    source = lock()
    cfg = plan()
    assert sha(PRETRAIN) == cfg["source_pretrained_sha256"]
    assert sha(cfg["parent_data_provenance"]) == cfg["parent_data_provenance_sha256"]
    out = ROOT / "prepared"
    out.mkdir(exist_ok=True)
    assert not (out / "teacher_complete.json").exists()
    shutil.copyfile(cfg["parent_data_provenance"], out / "data_provenance.json")
    provenance = {
        r["sample_id"]: r for r in json.loads((out / "data_provenance.json").read_text())["labels"]
    }
    assets = Assets()
    tr, _ = splits()
    m = initialized()
    m.requires_grad_(False)
    before = state_sha(m.state_dict())
    from dgmm_moe.reliability.gmm import targets

    yy, xx = torch.meshgrid(torch.arange(8, 256, 16), torch.arange(8, 256, 16), indexing="ij")
    ids = (yy * 256 + xx).flatten()
    xs = []
    masks = []
    anchors = []
    sum_a = 0.0
    count = 0
    available = 0
    tile_hashes = {}
    start = time.time()
    days0 = None
    for i, row in enumerate(tr):
        sid = row["sample_id"]
        sample = assets.load(row)
        bb = expand(sample)
        z = m(bb)["logits"].float()
        score, avail, _ = targets(z, bb)
        q = torch.where(avail, score, torch.ones_like(score))
        p = z.softmax(1)[0, 1].cpu()
        q = q.cpu()
        yp = sample["target"]
        q10 = F.interpolate(q[:, None], scale_factor=2, mode="nearest")[0, 0]
        a1 = q10 * (yp == 1) + (1 - q10) * p
        valid = yp != 255
        sum_a += float(a1[valid].double().sum())
        count += int(valid.sum())
        available += int(avail.sum())
        x, obs, days = temporal_read(assets, row)
        if days0 is None:
            days0 = days
        assert torch.equal(days0, days)
        select = valid.flatten()[ids] & obs[ids].any(1)
        xs.append(x[ids][select])
        masks.append(obs[ids][select])
        a = a1.flatten()[ids][select]
        anchors.append(torch.stack([1 - a, a], 1))
        f = out / "teacher" / (sid + ".pt")
        save(
            f,
            dict(
                q=q, teacher_positive=p, label_sha256=sha(assets.labels / sid / "pseudo_label.npy")
            ),
        )
        tile_hashes[sid] = dict(
            cache_sha256=sha(f),
            cube_sha256=sha(assets.cubes / sid / "x_cube.npy"),
            source_map_sha256=sha(assets.cubes / sid / "source_map.npy"),
            time_quality_sha256=sha(assets.cubes / sid / "time_quality.json"),
        )
        assert (
            tile_hashes[sid]["cube_sha256"] == provenance[sid]["cube_sha256"]
            and tile_hashes[sid]["source_map_sha256"] == provenance[sid]["source_map_sha256"]
        )
        if (i + 1) % 32 == 0:
            write(
                out / "progress.json",
                dict(
                    stage="fixed_teacher_and_observed_samples",
                    tiles=i + 1,
                    total=1296,
                    seconds=time.time() - start,
                ),
            )
    assert count > 0 and 0.01 < sum_a / count < 0.99 and (before == state_sha(m.state_dict()))
    data = dict(
        x=torch.cat(xs),
        observed=torch.cat(masks),
        anchor=torch.cat(anchors),
        days=days0,
        class_prior=torch.tensor([1 - sum_a / count, sum_a / count]),
    )
    assert (
        data["x"].isfinite().all()
        and data["observed"].any(1).all()
        and torch.allclose(data["anchor"].sum(1), torch.ones(len(data["anchor"])))
    )
    save(out / "samples.pt", data)
    write(
        out / "teacher_complete.json",
        dict(
            source_lock=source,
            teacher_sha256=cfg["teacher_checkpoint_sha256"],
            teacher_model_unchanged=True,
            sample_count=len(data["x"]),
            valid_train_pixels=count,
            class_prior=data["class_prior"].tolist(),
            gmm_available20=available,
            samples_sha256=sha(out / "samples.pt"),
            tile_hashes=tile_hashes,
            dates=days0.tolist(),
            train_reference_labels_read=False,
            seconds=time.time() - start,
        ),
    )


def fit_experts():
    cfg = plan()
    source = lock()
    done = json.loads((ROOT / "prepared/teacher_complete.json").read_text())
    assert (
        done["source_lock"] == source
        and sha(ROOT / "prepared/samples.pt") == done["samples_sha256"]
    )
    setup(cfg["experts"]["fit_seed"])
    s = torch.load(ROOT / "prepared/samples.pt", map_location="cpu", weights_only=False)
    x = s["x"]
    obs = s["observed"]
    a = s["anchor"]
    assert obs.any(1).all()
    observed = obs.float()[..., None]
    total = observed.sum(0).clamp_min(1)
    mean = (x * observed).sum(0) / total
    filled = torch.where(obs[..., None], x, mean[None])
    from sklearn.cluster import MiniBatchKMeans

    centers = []
    for c in (0, 1):
        km = MiniBatchKMeans(
            n_clusters=3, random_state=42030 + c, n_init=3, max_iter=100, batch_size=2048
        )
        km.fit(filled.flatten(1).numpy(), sample_weight=a[:, c].numpy())
        centers.append(torch.from_numpy(km.cluster_centers_).reshape(3, 26, 10))
    count = observed.sum((0, 1)).clamp_min(1)
    band_mean = (x * observed).sum((0, 1)) / count
    variance = ((x - band_mean).square() * observed).sum((0, 1)) / count
    scale = variance.sqrt().clamp(0.1, 3)
    e = Evidence(s["days"], torch.stack(centers), scale, s["class_prior"]).cuda()
    initial = e.templates.detach().cpu().clone()
    optimizer = torch.optim.Adam(e.parameters(), lr=0.01, weight_decay=0)
    n = len(x)
    history = []
    start = time.time()
    generator = torch.Generator().manual_seed(42030)
    x = x.cuda()
    obs = obs.cuda()
    a = a.cuda()
    for epoch in range(1, 31):
        order = torch.randperm(n, generator=generator)
        total_loss = 0.0
        updates = 0
        for ix0 in order.split(8192):
            ix = ix0.cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = e.fit_loss(x[ix], obs[ix], a[ix])
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(e.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ix)
            updates += 1
        history.append(
            dict(
                epoch=epoch,
                nll=total_loss / n,
                updates=updates,
                scales=e.scales().detach().cpu().tolist(),
                seconds=time.time() - start,
            )
        )
        write(ROOT / "prepared/expert_history.json", history)
        write(
            ROOT / "prepared/progress.json",
            dict(
                stage="expert_fit",
                epoch=epoch,
                total_epochs=30,
                train_nll=total_loss / n,
                seconds=time.time() - start,
            ),
        )
        print(json.dumps(history[-1]), flush=True)
    assert not torch.equal(initial, e.templates.detach().cpu())
    save(
        ROOT / "prepared/evidence.pt",
        dict(evidence=e.state_dict(), history=history, source_lock=source),
    )
    write(
        ROOT / "prepared/fit_complete.json",
        dict(
            source_lock=source,
            epochs=30,
            evidence_sha256=sha(ROOT / "prepared/evidence.pt"),
            sample_count=n,
            fit_reference_labels_read=False,
            seconds=time.time() - start,
        ),
    )


@torch.no_grad()
def evidence_tile(e, x, observed):
    return torch.cat(
        [
            e(values.cuda(), mask.cuda())["probability"][:, 1].cpu()
            for values, mask in zip(x.split(8192), observed.split(8192))
        ]
    )


@torch.no_grad()
def generate_targets():
    source = lock()
    prepared = json.loads((ROOT / "prepared/teacher_complete.json").read_text())
    assert prepared["source_lock"] == source
    assets = Assets()
    train_rows, _ = splits()
    evidence = load_evidence()
    hashes = {}
    valid_pixels = 0
    fallback_pixels = 0
    start = time.time()
    for index, row in enumerate(train_rows):
        sid = row["sample_id"]
        record = prepared["tile_hashes"][sid]
        file = ROOT / "prepared/teacher" / (sid + ".pt")
        assert sha(file) == record["cache_sha256"]
        assert sha(assets.cubes / sid / "x_cube.npy") == record["cube_sha256"]
        assert sha(assets.cubes / sid / "source_map.npy") == record["source_map_sha256"]
        assert sha(assets.cubes / sid / "time_quality.json") == record["time_quality_sha256"]
        cached = torch.load(file, map_location="cpu", weights_only=False)
        x, observed, days = temporal_read(assets, row)
        assert torch.equal(days, evidence.days.cpu())
        posterior = evidence_tile(evidence, x, observed)
        has_observation = observed.any(1)
        label_file = assets.labels / sid / "pseudo_label.npy"
        assert sha(label_file) == cached["label_sha256"]
        pseudo = torch.from_numpy(np.load(label_file, allow_pickle=False).astype(np.int64))
        q = F.interpolate(cached["q"][:, None], scale_factor=2, mode="nearest")[0, 0].flatten()
        target = soft_target(
            q, pseudo.flatten(), posterior, cached["teacher_positive"].flatten(), has_observation
        ).reshape(256, 256)
        assert target.isfinite().all() and (target >= 0).all() and (target <= 1).all()
        assert not target.requires_grad
        valid = pseudo != 255
        valid_pixels += int(valid.sum())
        fallback_pixels += int((valid.flatten() & ~has_observation).sum())
        destination = ROOT / "prepared/targets" / (sid + ".pt")
        save(destination, dict(positive=target))
        hashes[sid] = sha(destination)
        if (index + 1) % 32 == 0:
            write(
                ROOT / "prepared/progress.json",
                dict(
                    stage="soft_targets",
                    tiles=index + 1,
                    total=len(train_rows),
                    seconds=time.time() - start,
                ),
            )
    write(
        ROOT / "prepared/targets_complete.json",
        dict(
            source_lock=source,
            target_hashes=hashes,
            teacher_sha256=plan()["teacher_checkpoint_sha256"],
            evidence_sha256=sha(ROOT / "prepared/evidence.pt"),
            valid_pixels=valid_pixels,
            zero_observation_fallback=fallback_pixels,
            reference_labels_read=False,
            seconds=time.time() - start,
        ),
    )
