"""Generate reproducible Train/Validation/Test distribution diagnostics.

The script reads the Xinjiang 2021 cubes, pseudo labels, and region-level split
manifest. It samples a fixed set of spatial pixels from every cube for spectral
statistics and reads every label map for exact class-composition statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import zlib
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


SPLITS = ("train", "val", "test")
DISPLAY_NAMES = {"train": "Train", "val": "Validation", "test": "Test"}
COLORS = {"train": "#0072B2", "val": "#E69F00", "test": "#009E73"}
BAND_NAMES = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cube-root",
        type=Path,
        default=Path(r"E:\maize_paper_workspace\03_processed_data\cubes\xinjiang_2021\cubes\2021"),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path(
            r"E:\maize_paper_workspace\05_pseudo_labels\sam_refined"
            r"\xinjiang_vvp_mmi_confidence_corrected\conservative_v1"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            r"E:\maize_paper_workspace\01_code\spring_maize_paper_dataset"
            r"\generated\splits_2021_top5_region_holdout.csv"
        ),
    )
    parser.add_argument("--pixels-per-tile", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "region_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}: {path}")
    invalid = sorted({row["split"] for row in rows} - set(SPLITS))
    if invalid:
        raise ValueError(f"Unexpected splits: {invalid}")
    return rows


def deterministic_block(sample_id: str, count: int, side: int = 256) -> tuple[slice, slice]:
    block_side = int(np.sqrt(count))
    if block_side * block_side != count:
        raise ValueError("pixels_per_tile must be a perfect square for contiguous block sampling")
    if block_side > side:
        raise ValueError("sampling block cannot exceed the image side")
    seed = zlib.crc32(sample_id.encode("utf-8"))
    rng = np.random.default_rng(seed)
    top = int(rng.integers(0, side - block_side + 1))
    left = int(rng.integers(0, side - block_side + 1))
    return slice(top, top + block_side), slice(left, left + block_side)


def representative_sample(tile_stats: list[dict[str, float | str]]) -> str:
    supervised = np.array([float(item["supervised_fraction"]) for item in tile_stats])
    positive = np.array([float(item["positive_fraction"]) for item in tile_stats])
    center = np.array([np.median(supervised), np.median(positive)])
    scale = np.array([np.std(supervised), np.std(positive)])
    scale = np.maximum(scale, 1e-6)
    points = np.column_stack((supervised, positive))
    score = np.square((points - center) / scale).sum(axis=1)
    return str(tile_stats[int(np.argmin(score))]["sample_id"])


def collect_statistics(
    rows: list[dict[str, str]],
    cube_root: Path,
    label_root: Path,
    pixels_per_tile: int,
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    if pixels_per_tile <= 0 or pixels_per_tile > 256 * 256:
        raise ValueError("pixels_per_tile must lie in [1, 65536]")

    stats: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        stats[split] = {
            "sample_ids": [],
            "regions": set(),
            "label_counts": np.zeros(3, dtype=np.int64),
            "tile_stats": [],
            "ndvi": [],
            "band_sum": np.zeros(10, dtype=np.float64),
            "band_sum_sq": np.zeros(10, dtype=np.float64),
            "band_count": 0,
        }

    for index, row in enumerate(rows, start=1):
        split = row["split"]
        sample_id = row["sample_id"]
        split_stats = stats[split]
        split_stats["sample_ids"].append(sample_id)
        split_stats["regions"].add(row["region_id"])

        label_path = label_root / sample_id / "pseudo_label.npy"
        cube_path = cube_root / sample_id / "x_cube.npy"
        if not label_path.exists() or not cube_path.exists():
            raise FileNotFoundError(f"Missing cube or label for {sample_id}")

        label = np.load(label_path, mmap_mode="r")
        counts = np.array([(label == value).sum() for value in (0, 1, 255)], dtype=np.int64)
        split_stats["label_counts"] += counts
        valid = int(counts[0] + counts[1])
        split_stats["tile_stats"].append(
            {
                "sample_id": sample_id,
                "supervised_fraction": valid / label.size,
                "positive_fraction": int(counts[1]) / valid if valid else np.nan,
            }
        )

        cube = np.load(cube_path, mmap_mode="r")
        if cube.shape != (26, 10, 256, 256):
            raise ValueError(f"Unexpected cube shape for {sample_id}: {cube.shape}")
        row_slice, col_slice = deterministic_block(sample_id, pixels_per_tile)
        sampled = np.asarray(cube[:, :, row_slice, col_slice], dtype=np.float64)
        sampled = sampled.reshape(26, 10, pixels_per_tile)
        if sampled.shape != (26, 10, pixels_per_tile):
            raise RuntimeError(f"Unexpected sampled shape for {sample_id}: {sampled.shape}")
        if not np.isfinite(sampled).all():
            raise ValueError(f"Non-finite sampled reflectance for {sample_id}")

        red = sampled[:, 2]
        nir = sampled[:, 6]
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        split_stats["ndvi"].append(ndvi)
        split_stats["band_sum"] += sampled.sum(axis=(0, 2))
        split_stats["band_sum_sq"] += np.square(sampled).sum(axis=(0, 2))
        split_stats["band_count"] += sampled.shape[0] * sampled.shape[2]

        if index % 100 == 0 or index == len(rows):
            print(f"[scan] {index}/{len(rows)} samples", flush=True)

    representatives: dict[str, str] = {}
    for split in SPLITS:
        split_stats = stats[split]
        split_stats["regions"] = sorted(split_stats["regions"])
        split_stats["ndvi"] = np.concatenate(split_stats["ndvi"], axis=1)
        count = int(split_stats["band_count"])
        band_mean = split_stats["band_sum"] / count
        band_var = split_stats["band_sum_sq"] / count - np.square(band_mean)
        split_stats["band_mean"] = band_mean
        split_stats["band_std"] = np.sqrt(np.maximum(band_var, 0.0))
        representatives[split] = representative_sample(split_stats["tile_stats"])
    return stats, representatives


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
        }
    )


def plot_distribution(stats: dict[str, dict[str, object]], output_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8))
    labels = [DISPLAY_NAMES[split] for split in SPLITS]
    colors = [COLORS[split] for split in SPLITS]

    ax = axes[0, 0]
    sample_counts = [len(stats[split]["sample_ids"]) for split in SPLITS]
    bars = ax.bar(labels, sample_counts, color=colors, width=0.62)
    total = sum(sample_counts)
    for bar, value in zip(bars, sample_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, f"{value}\n({value/total:.1%})", ha="center", fontsize=8)
    ax.set_ylabel("Full 256x256 tiles")
    ax.set_title("(a) Region-level split size")
    ax.set_ylim(0, max(sample_counts) * 1.22)

    ax = axes[0, 1]
    composition = np.stack([stats[split]["label_counts"] for split in SPLITS]).astype(float)
    composition /= composition.sum(axis=1, keepdims=True)
    bottoms = np.zeros(3)
    class_colors = ("#56B4E9", "#E69F00", "#B8B8B8")
    for class_index, (name, color) in enumerate(zip(("Non-maize", "Maize", "Ignore"), class_colors)):
        values = composition[:, class_index]
        ax.bar(labels, values, bottom=bottoms, color=color, width=0.62, label=name)
        for x, bottom, value in zip(range(3), bottoms, values):
            if value >= 0.08:
                ax.text(x, bottom + value / 2, f"{value:.1%}", ha="center", va="center", fontsize=7.5)
        bottoms += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pixel fraction")
    ax.set_title("(b) Pixel-label composition")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))

    ax = axes[0, 2]
    values = [
        [float(item["supervised_fraction"]) for item in stats[split]["tile_stats"]]
        for split in SPLITS
    ]
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.58)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Supervised pixel fraction")
    ax.set_title("(c) Per-tile label coverage")

    ax = axes[1, 0]
    values = [
        [float(item["positive_fraction"]) for item in stats[split]["tile_stats"]]
        for split in SPLITS
    ]
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.58)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Maize fraction among supervised pixels")
    ax.set_title("(d) Per-tile class balance")

    ax = axes[1, 1]
    time_points = np.arange(1, 27)
    for split in SPLITS:
        ndvi = stats[split]["ndvi"]
        q25, median, q75 = np.quantile(ndvi, (0.25, 0.5, 0.75), axis=1)
        ax.plot(time_points, median, color=COLORS[split], label=DISPLAY_NAMES[split], linewidth=1.8)
        ax.fill_between(time_points, q25, q75, color=COLORS[split], alpha=0.13)
    ax.set_xlim(1, 26)
    ax.set_xticks((1, 5, 10, 15, 20, 26))
    ax.set_xlabel("Time point")
    ax.set_ylabel("NDVI median (IQR shading)")
    ax.set_title("(e) Temporal vegetation trajectory")
    ax.legend(loc="best")

    ax = axes[1, 2]
    band_x = np.arange(len(BAND_NAMES))
    for split in SPLITS:
        ax.plot(
            band_x,
            stats[split]["band_mean"],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            color=COLORS[split],
            label=DISPLAY_NAMES[split],
        )
    ax.set_xticks(band_x)
    ax.set_xticklabels(BAND_NAMES, rotation=35)
    ax.set_ylabel("Mean reflectance")
    ax.set_title("(f) Sampled spectral profile")
    ax.legend(loc="best")

    fig.suptitle("Xinjiang 2021 split-distribution diagnostics", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_split_distribution_preview.png")
    fig.savefig(output_dir / "fig_split_distribution_preview.pdf")
    plt.close(fig)


def make_rgb(cube: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    composite = np.median(cube[:, [2, 1, 0]], axis=0).transpose(1, 2, 0)
    scaled = (composite - low[None, None]) / np.maximum(high - low, 1e-6)[None, None]
    return np.clip(scaled, 0, 1) ** 0.82


def plot_representatives(
    stats: dict[str, dict[str, object]],
    representatives: dict[str, str],
    cube_root: Path,
    label_root: Path,
    output_dir: Path,
) -> None:
    apply_style()
    cubes = {
        split: np.load(cube_root / sample_id / "x_cube.npy")
        for split, sample_id in representatives.items()
    }
    composites = [np.median(cubes[split][:, [2, 1, 0]], axis=0).transpose(1, 2, 0) for split in SPLITS]
    pooled = np.concatenate([image.reshape(-1, 3) for image in composites], axis=0)
    low = np.quantile(pooled, 0.02, axis=0)
    high = np.quantile(pooled, 0.98, axis=0)

    label_cmap = ListedColormap(["#56B4E9", "#E69F00", "#B8B8B8"])
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.1))
    for column, split in enumerate(SPLITS):
        sample_id = representatives[split]
        rgb = make_rgb(cubes[split], low, high)
        label = np.load(label_root / sample_id / "pseudo_label.npy")
        display_label = np.full(label.shape, 2, dtype=np.uint8)
        display_label[label == 0] = 0
        display_label[label == 1] = 1

        tile_stat = next(item for item in stats[split]["tile_stats"] if item["sample_id"] == sample_id)
        axes[0, column].imshow(rgb)
        axes[0, column].set_title(f"{DISPLAY_NAMES[split]}\n{sample_id}", color=COLORS[split])
        axes[0, column].axis("off")
        axes[1, column].imshow(display_label, cmap=label_cmap, vmin=0, vmax=2, interpolation="nearest")
        axes[1, column].set_title(
            f"supervised={float(tile_stat['supervised_fraction']):.1%}, "
            f"maize={float(tile_stat['positive_fraction']):.1%}"
        )
        axes[1, column].axis("off")

    fig.text(0.018, 0.73, "Temporal-median RGB", rotation=90, va="center", fontsize=10, fontweight="bold")
    fig.text(0.018, 0.27, "Pseudo label", rotation=90, va="center", fontsize=10, fontweight="bold")
    fig.suptitle("Representative tiles nearest each split median", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.035, 0, 1, 0.96))
    fig.savefig(output_dir / "fig_split_representative_tiles.png")
    fig.savefig(output_dir / "fig_split_representative_tiles.pdf")
    plt.close(fig)


def write_summary(
    stats: dict[str, dict[str, object]],
    representatives: dict[str, str],
    output_path: Path,
) -> None:
    train_mean = np.asarray(stats["train"]["band_mean"])
    train_std = np.maximum(np.asarray(stats["train"]["band_std"]), 1e-6)
    output: dict[str, object] = {
        "sampling": "one deterministic random contiguous 8x8 spatial block per full tile"
    }
    for split in SPLITS:
        counts = np.asarray(stats[split]["label_counts"], dtype=np.int64)
        total = int(counts.sum())
        valid = int(counts[0] + counts[1])
        ndvi = np.asarray(stats[split]["ndvi"])
        spectral_shift = float(np.mean(np.abs(np.asarray(stats[split]["band_mean"]) - train_mean) / train_std))
        output[split] = {
            "sample_count": len(stats[split]["sample_ids"]),
            "regions": stats[split]["regions"],
            "label_counts": {"non_maize": int(counts[0]), "maize": int(counts[1]), "ignore": int(counts[2])},
            "supervised_fraction": valid / total,
            "maize_fraction_of_supervised": int(counts[1]) / valid,
            "median_ndvi_by_time_point": np.median(ndvi, axis=1).tolist(),
            "band_mean": np.asarray(stats[split]["band_mean"]).tolist(),
            "band_std": np.asarray(stats[split]["band_std"]).tolist(),
            "mean_absolute_standardized_spectral_shift_from_train": spectral_shift,
            "representative_sample": representatives[split],
        }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    stats, representatives = collect_statistics(
        rows,
        cube_root=args.cube_root,
        label_root=args.label_root,
        pixels_per_tile=args.pixels_per_tile,
    )
    plot_distribution(stats, args.output_dir)
    plot_representatives(stats, representatives, args.cube_root, args.label_root, args.output_dir)
    write_summary(stats, representatives, args.output_dir / "split_distribution_summary.json")
    print(json.dumps({"representatives": representatives}, indent=2), flush=True)


if __name__ == "__main__":
    main()
