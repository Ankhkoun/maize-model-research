"""Compute and freeze Train-only Xinjiang 2021 band statistics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.data.manifest import load_manifest, sha256_file
from src.data.normalization import compute_band_stats
from src.data.xinjiang_tiles import EXPECTED_BANDS, XinjiangTileDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=ROOT / "configs" / "paths.local.yaml")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "manifests" / "xinjiang_2021_e0_e1.csv",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manifests" / "xinjiang_2021_train_normalization.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.paths.open("r", encoding="utf-8") as stream:
        workspace_root = Path(yaml.safe_load(stream)["workspace_root"])
    records = load_manifest(args.manifest, workspace_root)
    train_records = [record for record in records if record.split == "train"]
    if len(train_records) != 495:
        raise ValueError(f"expected 495 Train tiles, got {len(train_records)}")
    dataset = XinjiangTileDataset(train_records, workspace_root)

    def progress(done: int, total: int) -> None:
        if done == 1 or done % 25 == 0 or done == total:
            print(f"normalization {done}/{total}", flush=True)

    stats = compute_band_stats(dataset, progress=progress)
    source_manifest = args.source_manifest or (
        workspace_root
        / "01_code"
        / "spring_maize_paper_dataset"
        / "generated"
        / "splits_2021_top5_region_holdout.csv"
    )
    document = {
        "schema_version": 1,
        "split": "train",
        "band_names": list(EXPECTED_BANDS),
        **stats,
        "manifest_sha256": sha256_file(args.manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(document, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
