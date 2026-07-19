# E0/E1 Test-only Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict, auditable Test-only evaluation entry and run each frozen E0/E1 checkpoint exactly once on the 305-tile Test split.

**Architecture:** Keep the training/Validation CLI unchanged and add a dedicated `scripts/evaluate_test.py`. Reuse the single full-tile evaluator through an explicit `expected_split` contract, and validate all frozen assets before constructing the Test dataset or executing inference.

**Tech Stack:** Python 3.11, PyTorch 2.10, pytest, YAML/JSON, Windows PowerShell, CUDA.

## Global Constraints

- Test may not influence tuning, thresholding, model/checkpoint selection, early stopping, or training.
- Test-only mode resolves exactly 305 Test records and no Train/Validation dataset.
- Preserve 24x24 windows, stride 24, full 256x256 stitching, `ignore_index=255`, and existing metric semantics.
- E0/E1 run in separate processes and fixed independent `test_evaluation` directories.
- Do not stage, commit, push, create a PR, or switch/create another branch.
- Report results as held-out pseudo-label agreement, not independent ground-truth accuracy.

---

### Task 1: Explicit split contract in the shared evaluator

**Files:**
- Modify: `src/training/evaluate.py`
- Test: `tests/test_full_tile_evaluation.py`

**Interfaces:**
- Consumes: dataset samples with `sample["split"]`.
- Produces: `evaluate_tiles(..., expected_split: str = "validation") -> dict[str, Any]`.

- [ ] Add a failing test that `expected_split="test"` accepts a Test tile and returns the same exact confusion metrics.
- [ ] Add a failing test that Test-only evaluation rejects a Validation tile with a `Test-only` error.
- [ ] Run `pytest tests/test_full_tile_evaluation.py -q` and confirm the new API test fails because `expected_split` is absent.
- [ ] Add minimal `expected_split` validation and replace Validation-specific error strings without changing metric computation.
- [ ] Re-run the focused test file and confirm all cases pass.

### Task 2: Strict Test dataset and frozen checkpoint loader

**Files:**
- Create: `scripts/evaluate_test.py`
- Create: `tests/test_test_evaluation_cli.py`

**Interfaces:**
- Produces: `build_test_dataset(manifest_path, workspace_root, normalization, expected_count=305)`.
- Produces: `load_frozen_checkpoint_for_test(checkpoint, model, config, manifest_sha256, normalization_sha256, map_location)` returning checkpoint metadata after loading only `payload["model"]`.
- Produces: `reserve_test_output_directory(checkpoint) -> Path` fixed to `<checkpoint-parent>/test_evaluation`.

- [ ] Write failing tests proving only Test records are resolved/constructed, the count guard is exact, a toy format-v1 checkpoint loads without optimizer objects, wrong hash/config/epoch are rejected, and an existing output directory is rejected.
- [ ] Run `pytest tests/test_test_evaluation_cli.py -q` and confirm collection fails because the module does not exist.
- [ ] Implement the smallest dedicated CLI and frozen asset table required by the tests.
- [ ] Re-run the new tests until green, then run both focused test files together.

### Task 3: CLI orchestration and pre-Test verification

**Files:**
- Modify: `scripts/evaluate_test.py`
- Test: `tests/test_test_evaluation_cli.py`

**Interfaces:**
- CLI inputs: `--config` and `--checkpoint`, with existing paths/manifest/normalization defaults.
- CLI outputs: fixed `test_evaluation/run_snapshot.json` and `test_evaluation/test_evaluation.json`.

- [ ] Add failing tests for exact frozen checkpoint selection and JSON envelope fields.
- [ ] Implement orchestration in this order: validate config/path/hash/output absence; load normalization; build exactly Test 305; construct model; load weights; evaluate with `expected_split="test"`; atomically write JSON.
- [ ] Run focused tests, then fresh full pytest.
- [ ] Run `scripts/smoke_e0.py` on CUDA in an ignored `.smoke` directory; confirm no formal Test records are loaded.

### Task 4: One-time formal E0 and E1 Test evaluation

**Files:**
- Create externally: each frozen run's `test_evaluation/` directory and two JSON files.

- [ ] Recompute checkpoint, manifest, and normalization SHA256 before the first run.
- [ ] Start a new process for E0 with the frozen E0 config/checkpoint; require exit 0 and `evaluated_tiles=305`.
- [ ] Without changing any setting, start a new process for E1 with the frozen E1 config/checkpoint; require exit 0 and `evaluated_tiles=305`.
- [ ] Validate supervised pixels equal the documented Test count and each confusion matrix sums to it.
- [ ] Recompute SHA256 for every output JSON and both checkpoints.
- [ ] Calculate E1-E0 Test deltas and list the existing Validation deltas alongside them.

### Task 5: Records, inventory, and final verification

**Files:**
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/STATUS.md`
- Modify: `docs/TASKS.md`
- Modify only if a new decision exists: `docs/DECISIONS.md`
- Create: `docs/handoffs/2026-07-19-e0-e1-test-evaluation-complete.md`
- Modify externally: `E:\maize_paper_workspace\00_docs\experiment_notes\...`
- Modify externally: `E:\maize_paper_workspace\00_docs\inventories\agent_project_handoff_2026-06-11.md`

- [ ] Record exact commands, hashes, metrics, confusion matrices, deltas, interpretation, and the no-Test-tuning statement.
- [ ] Refresh the formal workspace inventory with its fixed script and spot-check both result directories and new notes.
- [ ] Run fresh full pytest with a new basetemp, `python -m compileall -q src scripts tests`, and `git diff --check`.
- [ ] Audit `git status --short`, verify no stage/commit/push/PR action occurred, and hand off all changed/output paths.
