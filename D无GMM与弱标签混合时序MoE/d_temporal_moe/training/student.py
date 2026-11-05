"""Frozen teacher -> observed-only temporal evidence fit -> dense fixed targets -> FPN-only continuation."""

import json, time
import torch
from d_temporal_moe.runtime import ROOT, plan, lock, sha
from d_temporal_moe.io_utils import save, write, state_sha
from d_temporal_moe.data import Assets, splits, loader, batch, transform
from d_temporal_moe.training.initialization import student_initialized as initialized
from d_temporal_moe.training.common import dev
from d_temporal_moe.models.temporal_moe import soft_ce


def train():
    source = lock()
    done = json.loads((ROOT / "prepared/targets_complete.json").read_text())
    assert done["source_lock"] == source
    assets = Assets()
    tr, dv = splits()
    targets = {}
    for r in tr:
        f = ROOT / "prepared/targets" / (r["sample_id"] + ".pt")
        assert sha(f) == done["target_hashes"][r["sample_id"]]
        targets[r["sample_id"]] = torch.load(f, map_location="cpu", weights_only=False)["positive"]
    out = ROOT / "checkpoints"
    out.mkdir(parents=True, exist_ok=False)
    m = initialized()
    params = list(m.classifier.decoder.parameters())
    opt = torch.optim.AdamW(params, lr=0.00015, weight_decay=0.0001)
    frozen = state_sha(
        {k: v for k, v in m.classifier.state_dict().items() if not k.startswith("decoder.")}
    )
    history = []
    initial = dev(m, assets, dv)
    parent = torch.load(plan()["teacher_checkpoint"], map_location="cpu", weights_only=False)
    assert all((initial[k] == parent["metrics"][k] for k in ["tp", "fp", "fn", "tn"]))
    best = dict(epoch=0, metrics=initial)

    def payload(epoch, metrics):
        return dict(
            model=m.state_dict(),
            epoch=epoch,
            metrics=metrics,
            optimizer=opt.state_dict(),
            source_lock=source,
            plan=plan(),
        )

    save(out / "epoch00.pt", payload(0, initial))
    save(out / "best.pt", payload(0, initial))
    write(out / "best.json", best)
    start = time.time()
    for epoch in range(1, 13):
        m.train_mode()
        opt.zero_grad(set_to_none=True)
        s = 0.0
        updates = 0
        for step, b0 in enumerate(loader(tr, epoch + 100, True), 1):
            bb = batch(b0)
            target = transform(targets[b0["sample_id"]], b0["transform"])[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = m(bb)["logits"]
            loss = soft_ce(z, target, bb["valid"])
            (loss / 8).backward()
            s += float(loss.detach())
            if step % 8 == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                opt.step()
                opt.zero_grad(set_to_none=True)
                updates += 1
                if updates % 32 == 0:
                    write(
                        out / "progress.json",
                        dict(
                            epoch=epoch,
                            updates=updates,
                            total_method_updates=(epoch - 1) * 162 + updates,
                            total_updates=1944,
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
            continued_from_teacher=True,
        ),
    )


def seal():
    completed = json.loads((ROOT / "checkpoints/completed.json").read_text())
    assert completed["epochs"] == 12 and completed["source_lock"] == lock()
    assert completed["best_checkpoint_sha256"] == sha(ROOT / "checkpoints/best.pt")
    write(
        ROOT / "outputs/seal.json",
        dict(completed=completed, source_lock=lock(), selection_locked=True),
    )


def selected():
    from d_temporal_moe.checkpoints import restore_model

    record = json.loads((ROOT / "outputs/seal.json").read_text())
    assert record["source_lock"] == lock()
    cp = ROOT / "checkpoints/best.pt"
    assert sha(cp) == record["completed"]["best_checkpoint_sha256"]
    m = initialized()
    state = torch.load(cp, map_location="cpu", weights_only=False)
    restore_model(m, state)
    m.eval()
    m.requires_grad_(False)
    return (m, state, cp)
