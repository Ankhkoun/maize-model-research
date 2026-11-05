"""No target banks, model scores, fitted mixtures or learned teachers enter data."""

from pathlib import Path
import csv, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from d_temporal_moe.runtime import ASSETS, EXPECTED, sha
from d_temporal_moe.runtime import ROOT, path
from d_temporal_moe.io_utils import seed_int

MEAN = np.asarray(
    [
        4179.192015478227,
        4065.9106675194444,
        3957.274910960156,
        5207.452475253116,
        4327.12234687,
        4873.16102239,
        5049.1637925,
        5111.07806856,
        3056.86349163,
        2490.9675032,
    ],
    dtype=np.float32,
)
STD = np.asarray(
    [
        4041.5212325268735,
        3691.003119315892,
        3629.331318356375,
        2973.5178530908756,
        3569.73343885,
        3085.9151435,
        2937.56005119,
        2806.04462314,
        1808.30013156,
        1694.20220774,
    ],
    dtype=np.float32,
)


def rows(name):
    p = Path(ASSETS) / name
    assert sha(p) == EXPECTED[name]
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


def splits():
    train = [r for r in rows("train_validation500.csv") if r["split"] == "train"]
    dev = rows("gtdev60.csv")
    assert (
        len(train) == 1296
        and len(dev) == 60
        and (not {x["sample_id"] for x in train} & {x["sample_id"] for x in dev})
    )
    return (train, dev)


def transform(x, t):
    x = torch.rot90(x, t[0], (-2, -1))
    if t[1]:
        x = x.flip(-1)
    if t[2]:
        x = x.flip(-2)
    return x.contiguous()


class Assets:

    def __init__(self):
        self.cubes = path("cubes")
        self.labels = path("pseudo_labels")
        n = json.loads((Path(ASSETS) / "normalization_train1296.json").read_text())
        assert n["train_count"] == 1296
        self.mean = np.asarray(n["mean"], dtype=np.float32)[None, :, None, None]
        self.std = np.asarray(n["std"], dtype=np.float32)[None, :, None, None]
        provenance = ROOT / "prepared/data_provenance.json"
        self.label_hashes = (
            {
                x["sample_id"]: x["label_sha256"]
                for x in json.loads(provenance.read_text())["labels"]
            }
            if provenance.exists()
            else {}
        )

    def load(self, row, need_label=True, t=(0, False, False)):
        sid = row["sample_id"]
        p = self.cubes / sid
        raw = np.load(p / "x_cube.npy", allow_pickle=False)
        sm = np.load(p / "source_map.npy", allow_pickle=False)
        assert (
            raw.shape == (26, 10, 256, 256) and raw.dtype == np.float32 and np.isfinite(raw).all()
        )
        assert sm.shape == (26, 256, 256) and np.isin(sm, [1, 2, 3, 4]).all()
        meta = json.loads((p / "metadata.json").read_text())
        assert meta["sample_id"] == sid and meta["crs"] == "EPSG:3857"
        obs = torch.from_numpy(
            (sm == 1).reshape(26, 128, 2, 128, 2).mean((2, 4)).astype(np.float32)
        )
        obs = transform(obs, t)
        out = {
            "sample_id": sid,
            "region_id": row["region_id"],
            "agrifm_cube": transform(
                torch.from_numpy(
                    (raw * 10000 - MEAN[None, :, None, None]) / STD[None, :, None, None]
                ),
                t,
            ),
            "observed20": obs.gt(0).any(0),
            "transform": t,
            "metadata": meta,
        }
        if need_label:
            if sid in self.label_hashes:
                assert (
                    sha(self.labels / sid / "pseudo_label.npy") == self.label_hashes[sid]
                ), "training pseudo-label changed after preparation"
            y = np.load(self.labels / sid / "pseudo_label.npy", allow_pickle=False)
            assert y.shape == (256, 256) and np.isin(y, [0, 1, 255]).all()
            y = transform(torch.from_numpy(y.astype(np.int64)), t)
            out.update(target=y, valid=y != 255)
        return out

    def dev_target(self, sid):
        p = path("dev_labels_30m") / sid / "y_patch_30m.npy"
        if not p.exists():
            p = path("dev_labels_10m") / sid / "y_patch.npy"
        x = np.load(p, allow_pickle=False)
        if x.shape == (85, 85):
            return x
        assert x.shape == (256, 256)
        b = x[:255, :255].reshape(85, 3, 85, 3)
        v = b != 255
        n = v.sum((1, 3))
        pos = ((b == 1) & v).sum((1, 3))
        return np.where(n > 0, pos / np.maximum(n, 1) >= 0.5, 255).astype(np.uint8)


class Tiles(Dataset):

    def __init__(self, rows, epoch=0, augment=False):
        self.rows = rows
        self.epoch = epoch
        self.augment = augment
        self.assets = Assets()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        t = (0, False, False)
        if self.augment:
            g = torch.Generator().manual_seed(seed_int(42, self.epoch, row["sample_id"]))
            t = (
                int(torch.randint(0, 4, (), generator=g)),
                bool(torch.rand((), generator=g) < 0.5),
                bool(torch.rand((), generator=g) < 0.5),
            )
        return self.assets.load(row, True, t)


def collate(xs):
    assert len(xs) == 1
    x = xs[0]
    return {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in x.items()}


def loader(rows, epoch, shuffle):
    return DataLoader(
        Tiles(rows, epoch, shuffle),
        batch_size=1,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed_int(42, epoch, "loader")),
        multiprocessing_context="spawn",
    )


def batch(x, device="cuda"):
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in x.items()
    }
