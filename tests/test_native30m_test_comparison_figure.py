from pathlib import Path


def test_native30m_table_has_frozen_scope_and_correct_column_winners() -> None:
    from figures.gen_fig_native30m_test_comparison_table import (
        METHODS,
        METRICS,
        best_method_by_metric,
        validate_results,
    )

    assert METRICS == ("F1", "IoU", "Kappa", "Precision", "Recall", "OA")
    assert "Area ratio" not in METRICS
    assert tuple(METHODS) == (
        "Raw SAM + VVP/MMI Otsu pseudo-labels",
        "PEACENET EXP002",
        "E0 TSViT + DOY",
        "E1 TSViT + DOY + WPE",
        "E2-W TSViT + learned P_T + WPE",
    )
    assert best_method_by_metric() == {
        "F1": "E0 TSViT + DOY",
        "IoU": "E0 TSViT + DOY",
        "Kappa": "E0 TSViT + DOY",
        "Precision": "PEACENET EXP002",
        "Recall": "E0 TSViT + DOY",
        "OA": "E0 TSViT + DOY",
    }
    validate_results()


def test_native30m_table_exports_png_and_pdf(tmp_path: Path) -> None:
    from figures.gen_fig_native30m_test_comparison_table import render_table

    png_path, pdf_path = render_table(tmp_path)

    assert png_path == tmp_path / "fig_native30m_test_comparison_table.png"
    assert pdf_path == tmp_path / "fig_native30m_test_comparison_table.pdf"
    assert png_path.stat().st_size > 10_000
    assert pdf_path.stat().st_size > 1_000
