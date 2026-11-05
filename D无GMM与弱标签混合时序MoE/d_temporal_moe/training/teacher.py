"""Unweighted D teacher, shared by the two temporal-supervision variants."""

import json
import time
import torch
from torch.nn import functional as F
from d_temporal_moe.runtime import ROOT, plan, lock, sha
from d_temporal_moe.io_utils import state_sha, write, save
from d_temporal_moe.data import Assets, splits, loader, batch, transform
from d_temporal_moe.training.initialization import teacher_initialized as initialized
from d_temporal_moe.training.common import expand, dev


def joint(m, b, entry):
    y = transform(entry["labels"], b["transform"]).unsqueeze(0).cuda()
    w = transform(entry["weights"], b["transform"]).cuda()
    y = F.interpolate(y.float(), scale_factor=4, mode="nearest").long()
    w = F.interpolate(w.flatten(0, 1)[None], scale_factor=4, mode="nearest").reshape(
        1, 3, 3, 256, 256
    )
    labels = torch.cat([b["target"][:, None], y], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(b)["logits"]
    loss, posterior = m.module(logits.float(), labels, w)
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
    for epoch in range(1, 13):
        m.train_mode()
        dc.zero_grad(set_to_none=True)
        mc.zero_grad(set_to_none=True)
        s = 0.0
        updates = 0
        for step, b0 in enumerate(loader(tr, epoch + 100, True), 1):
            loss, _ = joint(m, batch(b0), cache[b0["sample_id"]])
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
