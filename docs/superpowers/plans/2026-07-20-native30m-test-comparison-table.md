# Native 30 m Test Comparison Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a reproducible publication-style table image comparing five methods on the same independent native 30 m Test protocol.

**Architecture:** Keep the frozen metrics as an explicit ordered data structure in one Matplotlib script. Expose small validation and best-method helpers for tests, then render a border-light table and export both PNG and PDF.

**Tech Stack:** Python 3.11, Matplotlib, pytest

## Global Constraints

- Methods: raw SAM+VVP/MMI Otsu pseudo labels, PEACENET EXP002, E0, E1, E2-W.
- Metrics and order: F1, IoU, Kappa, Precision, Recall, OA.
- Evaluation: 305 Test tiles and 2,203,625 native 30 m cells with `label != 255`.
- Display four decimals; bold the maximum in each metric column.
- Do not include Area ratio or upsampled 10 m results.
- Do not modify formal Test outputs or perform Git publication actions.

---

### Task 1: Reproducible comparison table

**Files:**
- Create: `tests/test_native30m_test_comparison_figure.py`
- Create: `figures/gen_fig_native30m_test_comparison_table.py`
- Create: `figures/fig_native30m_test_comparison_table.png`
- Create: `figures/fig_native30m_test_comparison_table.pdf`

**Interfaces:**
- Consumes: frozen native 30 m metrics recorded in `docs/EXPERIMENTS.md`.
- Produces: `METRICS`, `METHODS`, `best_method_by_metric()`, `validate_results()`, and `render_table(output_dir)`.

- [ ] **Step 1: Write failing tests** for method/metric order, absence of Area ratio, exact winning methods, validation, and PNG/PDF export.
- [ ] **Step 2: Run the focused test** and confirm it fails because `figures.gen_fig_native30m_test_comparison_table` does not exist.
- [ ] **Step 3: Implement the minimal Matplotlib module** with frozen values, validation, best-cell emphasis, and dual export.
- [ ] **Step 4: Run the focused test and full suite** using a worktree-local pytest `--basetemp`.
- [ ] **Step 5: Run the generator**, inspect the PNG, verify both outputs are non-empty, and run `git diff --check` plus whitespace checks for untracked text files.

Git staging, commits, pushes, and PR creation remain the user's responsibility and are intentionally excluded.
