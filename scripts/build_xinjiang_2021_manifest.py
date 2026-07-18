"""Build the repository-local Xinjiang 2021 E0/E1 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.data.manifest import build_repository_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=Path, default=ROOT / "configs" / "paths.local.yaml")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manifests" / "xinjiang_2021_e0_e1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.paths.open("r", encoding="utf-8") as stream:
        workspace_root = Path(yaml.safe_load(stream)["workspace_root"])
    source_manifest = args.source_manifest or (
        workspace_root
        / "01_code"
        / "spring_maize_paper_dataset"
        / "generated"
        / "splits_2021_top5_region_holdout.csv"
    )
    summary = build_repository_manifest(source_manifest, args.output, workspace_root)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": summary.counts,
                "source_sha256": summary.source_sha256,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
