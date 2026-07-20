"""Render the frozen five-method independent native-30 m Test comparison."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METRICS = ("F1", "IoU", "Kappa", "Precision", "Recall", "OA")

# All methods use the same 305 spatially held-out Test tiles and the same
# 2,203,625 annual-reference cells with label != 255. The two historical
# baselines were recomputed read-only on this support; E0/E1/E2-W are frozen
# one-time Test evaluations. Values are intentionally kept here so the figure
# is reproducible without reading or modifying formal Test assets.
METHODS: OrderedDict[str, dict[str, float]] = OrderedDict(
    [
        (
            "Raw SAM + VVP/MMI Otsu pseudo-labels",
            {
                "F1": 0.839264,
                "IoU": 0.723044,
                "Kappa": 0.720908,
                "Precision": 0.89823,
                "Recall": 0.78756,
                "OA": 0.86308,
            },
        ),
        (
            "PEACENET EXP002",
            {
                "F1": 0.863125,
                "IoU": 0.759208,
                "Kappa": 0.760399,
                "Precision": 0.91372,
                "Recall": 0.81784,
                "OA": 0.88227,
            },
        ),
        (
            "E0 TSViT + DOY",
            {
                "F1": 0.879612674,
                "IoU": 0.785096951,
                "Kappa": 0.782021820,
                "Precision": 0.891711023,
                "Recall": 0.867838221,
                "OA": 0.892183107,
            },
        ),
        (
            "E1 TSViT + DOY + WPE",
            {
                "F1": 0.870213437,
                "IoU": 0.770245872,
                "Kappa": 0.765665096,
                "Precision": 0.885267201,
                "Recall": 0.855663084,
                "OA": 0.884158149,
            },
        ),
        (
            "E2-W TSViT + learned P_T + WPE",
            {
                "F1": 0.877136474,
                "IoU": 0.781160358,
                "Kappa": 0.779779509,
                "Precision": 0.900506027,
                "Recall": 0.854949193,
                "OA": 0.891293664,
            },
        ),
    ]
)


def validate_results() -> None:
    """Validate the fixed comparison contract before rendering."""

    expected_methods = (
        "Raw SAM + VVP/MMI Otsu pseudo-labels",
        "PEACENET EXP002",
        "E0 TSViT + DOY",
        "E1 TSViT + DOY + WPE",
        "E2-W TSViT + learned P_T + WPE",
    )
    if tuple(METHODS) != expected_methods:
        raise ValueError("native-30 m comparison method order changed")
    if METRICS != ("F1", "IoU", "Kappa", "Precision", "Recall", "OA"):
        raise ValueError("native-30 m comparison metric contract changed")
    for method, values in METHODS.items():
        if tuple(values) != METRICS:
            raise ValueError(f"metric order mismatch for {method}")
        array = np.asarray(tuple(values.values()), dtype=np.float64)
        if not np.isfinite(array).all() or not ((0.0 <= array) & (array <= 1.0)).all():
            raise ValueError(f"invalid metric value for {method}")


def best_method_by_metric() -> dict[str, str]:
    """Return the unique maximum method for every displayed metric."""

    validate_results()
    return {
        metric: max(METHODS, key=lambda method: METHODS[method][metric])
        for metric in METRICS
    }


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render_table(output_dir: Path) -> tuple[Path, Path]:
    """Render the table and return the PNG and PDF paths."""

    validate_results()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    method_names = tuple(METHODS)
    cell_text = [
        [method, *(f"{METHODS[method][metric]:.4f}" for metric in METRICS)]
        for method in method_names
    ]
    columns = ("Method", *METRICS)

    fig, ax = plt.subplots(figsize=(11.8, 4.55))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        colLoc="center",
        cellLoc="center",
        colWidths=(0.39, 0.101, 0.101, 0.101, 0.101, 0.101, 0.101),
        bbox=(0.025, 0.18, 0.95, 0.64),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.4)

    header_fill = "#F2F4F7"
    rule_color = "#D9DEE5"
    model_fill = "#F7FAFE"
    best_row_fill = "#EDF5FF"
    text_color = "#1F2933"

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(rule_color)
        cell.set_linewidth(0.65)
        cell.visible_edges = "horizontal"
        cell.get_text().set_color(text_color)
        cell.set_height(0.112 if row else 0.102)
        if row == 0:
            cell.set_facecolor(header_fill)
            cell.get_text().set_fontweight("bold")
            cell.set_linewidth(0.85)
        elif row >= 3:
            cell.set_facecolor(best_row_fill if row == 3 else model_fill)
        else:
            cell.set_facecolor("#FFFFFF")
        if column == 0:
            cell.get_text().set_ha("left")
            cell.PAD = 0.03

    winners = best_method_by_metric()
    for metric_index, metric in enumerate(METRICS, start=1):
        winner_row = method_names.index(winners[metric]) + 1
        winner_cell = table[(winner_row, metric_index)]
        winner_cell.get_text().set_fontweight("bold")
        winner_cell.get_text().set_color("#123A63")

    fig.text(
        0.025,
        0.925,
        "Independent native 30 m Test comparison",
        fontsize=15.5,
        fontweight="bold",
        color="#17212B",
        ha="left",
    )
    fig.text(
        0.025,
        0.865,
        "Xinjiang 2021 · 305 spatially held-out Test tiles · 2,203,625 valid reference cells",
        fontsize=9.5,
        color="#586572",
        ha="left",
    )
    fig.text(
        0.025,
        0.085,
        "Best result in each column is bold. All metrics use the same label != 255 support without cropland or exclusion masking.",
        fontsize=8.8,
        color="#66727E",
        ha="left",
    )

    png_path = output_dir / "fig_native30m_test_comparison_table.png"
    pdf_path = output_dir / "fig_native30m_test_comparison_table.pdf"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    png_path, pdf_path = render_table(Path(__file__).resolve().parent)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
