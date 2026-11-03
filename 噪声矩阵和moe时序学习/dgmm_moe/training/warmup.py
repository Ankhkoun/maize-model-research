"""Three ordinary CE epochs, following the shared historical warmup schedule."""

import hashlib
import json
import numpy as np
import torch
from dgmm_moe.runtime import ROOT, lock, sha, require_inputs
from dgmm_moe.io_utils import setup, save, write, state_sha
from dgmm_moe.data import Assets, splits, loader, batch
from dgmm_moe.reliability.losses import ordinary_ce
from dgmm_moe.models.network import ClassifierModel
from dgmm_moe.training.common import dev


def check_data():
    require_inputs()
    train, development = splits()
    a = Assets()
    records = []
    for row in train:
        sid = row["sample_id"]
        p = a.labels / sid
        meta = json.loads((p / "metadata.json").read_text())
        encoded = json.dumps(meta).lower()
        assert meta["mode"] == "vvp_mmi_intersection" and meta["mmi_method"] == "otsu"
        assert "gmm" not in encoded and "moe" not in encoded
        y = np.load(p / "pseudo_label.npy", allow_pickle=False)
        assert y.shape == (256, 256) and np.isin(y, [0, 1, 255]).all()
        records.append(
            dict(
                sample_id=sid,
                label_sha256=sha(p / "pseudo_label.npy"),
                metadata_sha256=sha(p / "metadata.json"),
            )
        )
    write(
        ROOT / "prepared/data_provenance.json",
        dict(
            labels=records,
            train=1296,
            development=60,
            train_only=True,
            reference_labels_read=False,
            source_lock=lock(),
        ),
    )
    print("Train weak-label provenance checked; no Dev/Test reference labels read")


def train():
    require_inputs()
    out = ROOT / "prepared"
    if (out / "warmup.pt").exists():
        raise FileExistsError("A CE3 checkpoint already exists")
    check_data()
    setup(42)
    m = ClassifierModel().cuda()
    params = list(m.classifier.decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.00015, weight_decay=0.0001)
    train_rows, development = splits()
    assets = Assets()
    frozen = state_sha(
        {k: v for k, v in m.classifier.state_dict().items() if not k.startswith("decoder.")}
    )
    history = []
    for epoch in range(1, 4):
        m.train_mode()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        trace = hashlib.sha256()
        updates = 0
        for step, b0 in enumerate(loader(train_rows, epoch, True), 1):
            b = batch(b0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = m(b)["logits"]
            loss = ordinary_ce(z, b)
            (loss / 8).backward()
            trace.update(str((b0["sample_id"], b0["transform"])).encode())
            losses.append(float(loss.detach()))
            if step % 8 == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        assert updates == 162
        assert frozen == state_sha(
            {k: v for k, v in m.classifier.state_dict().items() if not k.startswith("decoder.")}
        )
        evaluation = dev(m, assets, development)
        history.append(
            dict(
                epoch=epoch,
                loss=float(np.mean(losses)),
                updates=updates,
                metrics=evaluation,
                augmentation_sha256=trace.hexdigest(),
            )
        )
        write(out / "warmup_history.json", history)
        save(
            out / "warmup_last.pt",
            dict(
                classifier=m.classifier.state_dict(),
                optimizer=optimizer.state_dict(),
                history=history,
                source_lock=lock(),
            ),
        )
        print(json.dumps(dict(epoch=epoch, dev=evaluation["miou"])), flush=True)
    digest = state_sha(m.classifier.state_dict())
    save(
        out / "warmup.pt",
        dict(
            classifier=m.classifier.state_dict(),
            classifier_sha256=digest,
            history=history,
            initialization="official_AgriFM_fresh_decoder_classifier_only_warmup",
            source_lock=lock(),
        ),
    )
    write(
        out / "warmup_complete.json",
        dict(
            status="completed",
            epochs=3,
            updates=486,
            classifier_sha256=digest,
            checkpoint_sha256=sha(out / "warmup.pt"),
            source_lock=lock(),
        ),
    )
