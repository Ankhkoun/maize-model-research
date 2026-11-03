"""D+GMM training: detached GMM weights scale the three-view joint NLL. Historical 12-epoch schedule preserved."""

import json
import time
import torch
from torch.nn import functional as F
from dgmm_moe.runtime import ROOT, plan, lock, sha
from dgmm_moe.io_utils import state_sha, write, save
from dgmm_moe.data import Assets, splits, loader, batch, transform
from dgmm_moe.training.initialization import teacher_initialized as initialized
from dgmm_moe.training.common import expand, dev


def joint(m, b, entry, gmm, epoch):
    y = transform(entry["labels"], b["transform"]).unsqueeze(0).cuda()
    w = transform(entry["weights"], b["transform"]).cuda()
    y = F.interpolate(y.float(), scale_factor=4, mode="nearest").long()
    w = F.interpolate(w.flatten(0, 1)[None], scale_factor=4, mode="nearest").reshape(
        1, 3, 3, 256, 256
    )
    labels = torch.cat([b["target"][:, None], y], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(b)["logits"]
    patch = torch.where(
        gmm["available"], 0.05 + 0.95 * gmm["score"].float(), torch.ones_like(gmm["score"])
    )
    pixel_weight = F.interpolate(patch[:, None], size=logits.shape[-2:], mode="nearest")[
        :, 0
    ].detach()
    loss, posterior = m.module(logits.float(), labels, w, pixel_weight, min(1.0, epoch / 3))
    return (loss, posterior)


def bank():
    complete = json.loads((ROOT / "prepared/complete.json").read_text())
    assert complete["source_lock"] == lock()
    parent = ROOT
    tr, _ = splits()
    hashes = complete["neighbor_hashes"]
    cache = {}
    for r in tr:
        f = parent / "prepared/neighbors" / (r["sample_id"] + ".pt")
        assert sha(f) == hashes[r["sample_id"]]
        cache[r["sample_id"]] = torch.load(f, map_location="cpu", weights_only=False)
    return cache


@torch.no_grad()
def refresh(m, assets, rs, out, epoch):
    from dgmm_moe.reliability.gmm import targets
    from dgmm_moe.reliability.losses import eligible

    m.eval()
    before = state_sha(m.state_dict())
    cache = {}
    available = eligible_count = 0
    sums = [0.0, 0.0]
    counts = [0, 0]
    for row in rs:
        bb = expand(assets.load(row))
        logits = m(bb)["logits"].float()
        score, avail, modes = targets(logits.float(), bb)
        valid, classes, _ = eligible(bb)
        eligible_count += int(valid.sum())
        available += int(avail.sum())
        for c in (0, 1):
            mask = avail & (classes == c)
            sums[c] += float(score[mask].sum())
            counts[c] += int(mask.sum())
        cache[row["sample_id"]] = dict(score=score.cpu(), available=avail.cpu())
    assert state_sha(m.state_dict()) == before
    save(out / f"gmm_cache{epoch:02}.pt", cache)
    write(
        out / f"gmm_refresh{epoch:02}.json",
        dict(
            epoch=epoch,
            eligible=eligible_count,
            available=available,
            mean_q_by_class=[sums[c] / max(1, counts[c]) for c in (0, 1)],
            available_by_class=counts,
            source="current_classifier_unweighted_CE",
            source_lock=lock(),
        ),
    )
    return cache


def train():
    source = lock()
    out = ROOT / "checkpoints"
    out.mkdir(parents=True, exist_ok=False)
    m = initialized()
    assets = Assets()
    tr, dv = splits()
    cache = bank()
    a = list(m.classifier.decoder.parameters())
    b = list(m.module.parameters())
    dc = torch.optim.AdamW(a, lr=0.00015, weight_decay=0.0001)
    mc = torch.optim.AdamW(b, lr=0.0001, weight_decay=0.0001)
    frozen = state_sha(
        {k: v for k, v in m.classifier.state_dict().items() if not k.startswith("decoder.")}
    )
    history = []
    initial = dev(m, assets, dv)
    best = dict(epoch=0, metrics=initial)

    def payload(epoch, metrics):
        return dict(
            model=m.state_dict(),
            epoch=epoch,
            metrics=metrics,
            decoder_optimizer=dc.state_dict(),
            noise_optimizer=mc.state_dict(),
            source_lock=source,
            plan=plan(),
        )

    save(out / "epoch00.pt", payload(0, initial))
    save(out / "best.pt", payload(0, initial))
    write(out / "best.json", best)
    start = time.time()
    gmm_cache = None
    for epoch in range(1, 13):
        if epoch in (1, 4, 7, 10):
            gmm_cache = refresh(m, assets, tr, out, epoch)
        m.train_mode()
        dc.zero_grad(set_to_none=True)
        mc.zero_grad(set_to_none=True)
        s = 0.0
        updates = 0
        for step, b0 in enumerate(loader(tr, epoch + 100, True), 1):
            gm = {
                k: transform(v[0], b0["transform"]).unsqueeze(0).cuda()
                for k, v in gmm_cache[b0["sample_id"]].items()
            }
            loss, _ = joint(m, batch(b0), cache[b0["sample_id"]], gm, epoch)
            (loss / 8).backward()
            s += float(loss.detach())
            if step % 8 == 0:
                torch.nn.utils.clip_grad_norm_(a, 1.0, error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(b, 1.0, error_if_nonfinite=True)
                dc.step()
                mc.step()
                dc.zero_grad(set_to_none=True)
                mc.zero_grad(set_to_none=True)
                updates += 1
                if updates % 32 == 0:
                    write(
                        out / "progress.json",
                        dict(
                            epoch=epoch,
                            updates=updates,
                            total_method_updates=(epoch - 1) * 162 + updates,
                            seconds=time.time() - start,
                        ),
                    )
        assert updates == 162 and frozen == state_sha(
            {k: v for k, v in m.classifier.state_dict().items() if not k.startswith("decoder.")}
        )
        evaluation = dev(m, assets, dv)
        history.append(
            dict(
                epoch=epoch,
                updates=updates,
                metrics=evaluation,
                loss=s / 1296,
                matrices=m.module.matrices().detach().cpu().tolist(),
                seconds=time.time() - start,
            )
        )
        save(out / f"epoch{epoch:02}.pt", payload(epoch, evaluation))
        save(out / "last.pt", payload(epoch, evaluation))
        write(out / "history.json", history)
        if evaluation["miou"] > best["metrics"]["miou"]:
            best = dict(epoch=epoch, metrics=evaluation)
            save(out / "best.pt", payload(epoch, evaluation))
            write(out / "best.json", best)
        write(
            out / "progress.json",
            dict(
                epoch=epoch,
                updates=162,
                total_method_updates=epoch * 162,
                best_epoch=best["epoch"],
                best_dev_miou=best["metrics"]["miou"],
                seconds=time.time() - start,
            ),
        )
        print(
            json.dumps(dict(epoch=epoch, dev=evaluation["miou"], best_epoch=best["epoch"])),
            flush=True,
        )
    write(
        out / "completed.json",
        dict(
            status="completed",
            epochs=12,
            updates=1944,
            best=best,
            best_checkpoint_sha256=sha(out / "best.pt"),
            source_lock=source,
        ),
    )
