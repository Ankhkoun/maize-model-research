# E2-W Frozen Test Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze E2-W epoch 13, independently replay Validation, and run exactly one pseudo-label Test evaluation plus one independent 30m ground-truth Test evaluation using the same protocol as E0/E1.

**Architecture:** Extend the existing shared `FrozenTestAsset` registry with one E2-W entry; both Test CLIs already consume that registry and therefore need no separate model-loading path. Preserve the Validation-only training boundary, prove the frozen checkpoint before Test, and use the existing one-time output-directory guards for both Test runs.

**Tech Stack:** Python 3.11, PyTorch 2.10.0+cu128, pytest, YAML, CUDA on RTX 5060 Ti, PowerShell.

## Global Constraints

- E2-W is frozen at epoch 13 with Validation maize IoU `0.9369115428555538` and checkpoint SHA256 `A74C8A33030172E94020410D1E46FB3439C4438ECC1E796F06CEA347DF859428`.
- Test 305 may be read only after the independent Validation replay matches the frozen checkpoint record.
- Test must not change the model, checkpoint, threshold, early stopping, configuration, normalization, manifest, or selection decision.
- Pseudo-label Test and ground-truth Test each run exactly once in an independent process.
- Native 30m is the primary independent-reference result; upsampled10m is an auxiliary label-support result.
- Preserve all existing E0/E1 frozen assets and results.
- Do not stage, commit, push, switch/create branches, or create a PR; the user owns Git publication.

---

### Task 1: Add the E2-W frozen Test asset with TDD

**Files:**
- Modify: `tests/test_test_evaluation_portability.py`
- Modify: `scripts/evaluate_test.py`
- Modify: `scripts/evaluate_test_ground_truth.py`

**Interfaces:**
- Consumes: `FrozenTestAsset(experiment_id, checkpoint, sha256, epoch, validation_maize_iou)` and `FROZEN_TEST_ASSETS`.
- Produces: `FROZEN_TEST_ASSETS["E2W"]`, automatically shared by both Test CLIs through `load_frozen_checkpoint_for_test`.

- [ ] **Step 1: Write the failing frozen-asset test**

Add to `tests/test_test_evaluation_portability.py`:

```python
def test_e2w_formal_frozen_asset_is_exact() -> None:
    asset = FROZEN_TEST_ASSETS["E2W"]

    assert asset.experiment_id == "E2W"
    assert asset.checkpoint == Path(
        "06_models/retrain_outputs/maize_model_research/"
        "e2w_tsvit_pt_mexican_hat_k5_seed42/best.pt"
    )
    assert asset.sha256 == (
        "A74C8A33030172E94020410D1E46FB3439C4438ECC1E796F06CEA347DF859428"
    )
    assert asset.epoch == 13
    assert asset.validation_maize_iou == 0.9369115428555538
```

- [ ] **Step 2: Run RED and verify the expected failure**

Run:

```powershell
D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_test_evaluation_portability.py::test_e2w_formal_frozen_asset_is_exact --basetemp=.smoke/pytest-e2w-test-asset-red
```

Expected: FAIL with `KeyError: 'E2W'` because the frozen registry does not yet contain E2-W.

- [ ] **Step 3: Add the minimal production entry**

Add to `FROZEN_TEST_ASSETS` in `scripts/evaluate_test.py`:

```python
    "E2W": FrozenTestAsset(
        experiment_id="E2W",
        checkpoint=Path(
            "06_models/retrain_outputs/maize_model_research/"
            "e2w_tsvit_pt_mexican_hat_k5_seed42/best.pt"
        ),
        sha256="A74C8A33030172E94020410D1E46FB3439C4438ECC1E796F06CEA347DF859428",
        epoch=13,
        validation_maize_iou=0.9369115428555538,
    ),
```

Change only user-facing module/runtime text from “E0/E1” to “E0/E1/E2-W” in the two evaluation scripts. Do not alter loaders, metric computation, or output paths.

- [ ] **Step 4: Run GREEN and Test-entry regressions**

Run:

```powershell
D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_test_evaluation_portability.py tests/test_test_evaluation_cli.py tests/test_test_evaluation_training_guard.py tests/test_test_evaluation_e0_training_compatibility.py tests/test_test_evaluation_main.py tests/test_test_ground_truth_cli.py --basetemp=.smoke/pytest-e2w-test-asset-green
```

Expected: all selected tests PASS with no failures.

---

### Task 2: Independently replay and freeze E2-W Validation

**Files:**
- Read: `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42\best.pt`
- Create: `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42\validation_replay_best_epoch13_20260719\validation_replay.json`

**Interfaces:**
- Consumes: existing `scripts/train_e0.py --validation-only CHECKPOINT`.
- Produces: a new-process Validation replay document whose metrics must exactly match the epoch 13 training record before Test is authorized.

- [ ] **Step 1: Verify frozen preconditions without Test**

Run `Get-FileHash` for `best.pt`, verify the output directory does not already exist, verify `test_evaluation` and `test_evaluation_ground_truth` do not exist, and confirm no training process is running. Expected checkpoint hash is exactly `A74C8A33030172E94020410D1E46FB3439C4438ECC1E796F06CEA347DF859428`.

- [ ] **Step 2: Run independent full Validation replay**

Run in a new process:

```powershell
D:\Anaconda3\envs\cawa\python.exe -u scripts/train_e0.py --paths D:\cj_swcc\maize-model-research\configs\paths.local.yaml --manifest manifests\xinjiang_2021_e0_e1.csv --normalization manifests\xinjiang_2021_train_normalization.json --config configs\models\tsvit_e2w_pt_mexican_hat_k5.yaml --output-dir E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42\validation_replay_best_epoch13_20260719 --physical-batch-size 16 --validation-only E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42\best.pt
```

Expected: exit 0; evaluated tiles `276`; supervised pixels `11568302`; maize IoU `0.9369115428555538`; F1 `0.9674283230036226`; Kappa `0.9228124899929513`; confusion matrix `[[4663806,213474],[222120,6468902]]`.

- [ ] **Step 3: Freeze and audit the replay output**

Compute SHA256 of `validation_replay.json`. Verify confusion-matrix sum equals `11568302`, all metrics are finite, and no `test_evaluation*` directory was created. If any expected field differs, stop before Test and use systematic debugging.

---

### Task 3: Run the pre-Test verification gate

**Files:**
- Verify: `src`, `scripts`, `tests`, and all current E2-W worktree changes.

**Interfaces:**
- Consumes: the E2-W implementation and Task 1 frozen registry.
- Produces: fresh evidence that the codebase passes before first Test access.

- [ ] **Step 1: Run the full test suite**

```powershell
D:\Anaconda3\envs\cawa\python.exe -m pytest -q --basetemp=.smoke/pytest-e2w-pretest-full
```

Expected: exit 0 and zero failed tests.

- [ ] **Step 2: Run compilation and diff checks**

```powershell
D:\Anaconda3\envs\cawa\python.exe -m compileall -q src scripts tests
git diff --check
```

Expected: both commands exit 0; CRLF conversion notices are acceptable, whitespace errors are not.

- [ ] **Step 3: Recheck immutable Test outputs**

Confirm both E2-W output directories are absent immediately before the first Test process. If either exists, stop instead of deleting or overwriting it.

---

### Task 4: Run the one-time pseudo-label Test evaluation

**Files:**
- Create: `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42\test_evaluation\run_snapshot.json`
- Create: `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42\test_evaluation\test_evaluation.json`

**Interfaces:**
- Consumes: `scripts/evaluate_test.py`, the frozen E2-W asset, Test 305 pseudo-labels.
- Produces: the same pseudo-label agreement metrics schema as E0/E1.

- [ ] **Step 1: Launch exactly one independent Test process**

```powershell
D:\Anaconda3\envs\cawa\python.exe -u scripts/evaluate_test.py --paths D:\cj_swcc\maize-model-research\configs\paths.local.yaml --manifest manifests\xinjiang_2021_e0_e1.csv --normalization manifests\xinjiang_2021_train_normalization.json --config configs\models\tsvit_e2w_pt_mexican_hat_k5.yaml --checkpoint E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42\best.pt
```

Expected: exit 0 and one JSON result on stdout. Do not rerun after a successful process.

- [ ] **Step 2: Audit the pseudo-label result**

Verify `experiment_id == "E2W"`, `evaluation_split == "test"`, checkpoint SHA/epoch/Validation IoU match the frozen asset, `evaluated_tiles == 305`, all metrics are finite, and confusion-matrix sum equals `supervised_pixels`. Compute SHA256 for both output JSON files and compare IoU/F1/Kappa to the frozen E0 Test results without changing any settings.

---

### Task 5: Run the one-time independent 30m ground-truth Test

**Files:**
- Create: `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42\test_evaluation_ground_truth\run_snapshot.json`
- Create: `...\test_evaluation_ground_truth\native30m\test_evaluation.json`
- Create: `...\test_evaluation_ground_truth\upsampled10m\test_evaluation.json`

**Interfaces:**
- Consumes: `scripts/evaluate_test_ground_truth.py`, the same frozen model, Test 305 cubes, and `y_patch_30m.npy` labels.
- Produces: native30m primary metrics and upsampled10m auxiliary metrics from one model-inference pass.

- [ ] **Step 1: Launch exactly one independent ground-truth process**

```powershell
D:\Anaconda3\envs\cawa\python.exe -u scripts/evaluate_test_ground_truth.py --paths D:\cj_swcc\maize-model-research\configs\paths.local.yaml --manifest manifests\xinjiang_2021_e0_e1.csv --normalization manifests\xinjiang_2021_train_normalization.json --config configs\models\tsvit_e2w_pt_mexican_hat_k5.yaml --checkpoint E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42\best.pt
```

Expected: exit 0 and both scales in one JSON object on stdout. Do not rerun after success.

- [ ] **Step 2: Audit both independent-reference results**

Verify each scale has `evaluated_tiles == 305`, finite metrics, exact checkpoint provenance, and confusion-matrix conservation. Native30m must have `supervised_pixels == 2203625`; upsampled10m must have `supervised_pixels == 19832625`. Compute output SHA256 values and compare native30m IoU/F1/Kappa to E0 without changing any setting.

---

### Task 6: Record results and run final verification

**Files:**
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/STATUS.md`
- Modify: `docs/TASKS.md`
- Create: `docs/handoffs/2026-07-19-e2w-test-evaluation-complete.md`
- Create: `E:\maize_paper_workspace\00_docs\experiment_notes\maize_model_research_e2w_test_evaluation_2026-07-19.md`
- Update: formal workspace inventory using its fixed inventory script.

**Interfaces:**
- Consumes: frozen Validation replay and all three Test result documents.
- Produces: reproducible experiment registry, status, task completion, handoff, workspace note, and refreshed inventory.

- [ ] **Step 1: Record exact provenance and metrics**

Document the checkpoint SHA/epoch, Validation replay SHA and full metrics, pseudo-label Test SHA/metrics/confusion matrix, native30m and upsampled10m SHA/metrics/confusion matrices, E2-W minus E0 differences, interpretation boundaries, commands, and confirmation that Test was not used for selection.

- [ ] **Step 2: Refresh the formal workspace inventory**

Read and follow `E:\maize_paper_workspace\00_docs\inventories\AGENT_FIXED_RULES.md`, use the fixed inventory refresh command, and audit that every new E2-W result/note appears exactly once. Do not delete or overwrite formal assets.

- [ ] **Step 3: Run fresh completion verification**

```powershell
D:\Anaconda3\envs\cawa\python.exe -m pytest -q --basetemp=.smoke/pytest-e2w-posttest-full
D:\Anaconda3\envs\cawa\python.exe -m compileall -q src scripts tests
git diff --check
git status --short
```

Expected: full pytest exit 0 with zero failures; compileall and diff check exit 0; git status shows only intended E2-W implementation/spec/plan/result-document changes and pre-existing E2-W worktree changes. Do not stage or commit them.
