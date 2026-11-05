"""Atomic local writes, lossless tensor fingerprints, and historical seed setup."""

from pathlib import Path
import hashlib
import json
import os
import random
import tempfile
import numpy as np
import torch


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write(path, data):
    _atomic(
        path,
        lambda p: p.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        ),
    )


def save(path, data):
    from d_temporal_moe.checkpoints import pack

    payload = pack(data)

    def serialize(temporary):
        with temporary.open("wb") as stream:
            torch.save(payload, stream)

    _atomic(path, serialize)


def state_sha(d):
    h = hashlib.sha256()
    for n, t in sorted(d.items()):
        h.update(n.encode())
        h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def seed_int(seed, epoch, s):
    return int.from_bytes(hashlib.sha256(f"{seed}|{epoch}|{s}".encode()).digest()[:8], "little") % (
        2**63 - 1
    )


def setup(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
