import re, json, datetime
import numpy as np
import torch


def read(assets, row):
    p = assets.cubes / row["sample_id"]
    raw = np.load(p / "x_cube.npy", allow_pickle=False)
    source = np.load(p / "source_map.npy", allow_pickle=False)
    meta = json.loads((p / "metadata.json").read_text())
    assert meta["band_names"] == ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    assert (
        raw.shape == (26, 10, 256, 256)
        and source.shape == (26, 256, 256)
        and np.isfinite(raw).all()
    )
    quality = json.loads((p / "time_quality.json").read_text())
    assert len(quality) == 26
    days = []
    for i, q in enumerate(quality):
        assert q["slot_id"] == f"t{i + 1:02d}"
        a = datetime.date(2021, 4, 1) + datetime.timedelta(days=7 * i)
        b = datetime.date(2021, 9, 30) if i == 25 else a + datetime.timedelta(days=6)
        match = re.search("_t\\d{2}_(\\d{8})_(\\d{8})\\.tif$", q.get("path") or "")
        if match:
            assert tuple(
                (datetime.datetime.strptime(s, "%Y%m%d").date() for s in match.groups())
            ) == (a, b)
        days.append((a.timetuple().tm_yday + b.timetuple().tm_yday) / 2)
    x = (raw - assets.mean) / assets.std
    observed = source == 1
    x = np.where(observed[:, None], x, 0).astype(np.float32)
    return (
        torch.from_numpy(x.transpose(2, 3, 0, 1).reshape(-1, 26, 10).copy()),
        torch.from_numpy(observed.transpose(1, 2, 0).reshape(-1, 26).copy()),
        torch.tensor(days),
    )
