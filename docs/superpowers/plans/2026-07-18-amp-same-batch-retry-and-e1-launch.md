# AMP Same-Batch Retry And E1 Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add auditable AMP same-effective-batch loss-scale backoff, validate the frozen E1 configuration on real CUDA data, and launch formal Xinjiang 2021 E1 training without loading Test.

**Architecture:** `E0Trainer` retains the existing tile/group/microbatch hierarchy, but delegates one effective group to a retry loop that owns zeroing, backward, gradient inspection, scale backoff, audit logging, and the single successful optimizer/scheduler/global-step transition. The existing formal CLI is generalized to validate either E0 or E1 and select an experiment-specific output directory while the dataset builder remains restricted to Train and Validation.

**Tech Stack:** Python 3.11, PyTorch 2.10 CUDA AMP, pytest, PyYAML, JSON Lines, PowerShell process launch.

## Global Constraints

- Use `D:\Anaconda3\envs\cawa\python.exe`; do not install or upgrade dependencies.
- Preserve the dirty worktree and do not stage, commit, push, create a PR, reset, clean, or edit the external Exact/TimeMIL trees.
- Formal data remain Xinjiang 2021 Train 495 / Validation 276; no code path may resolve or instantiate Test during training or smoke.
- E1 differs from completed E0 only by enabled WPE and its parameters; use `24x24`, patch 2, batch 16/16, seed 42, AdamW, the frozen schedule, and Validation maize IoU selection.
- Freeze AMP policy at init scale 8192, backoff factor 0.5, minimum scale 128, at most 6 backoffs per effective batch, and growth interval 1,000,000.
- Never skip a supervised batch, advance training state after a failed attempt, switch the failed batch to FP32, or apply gradient clipping.

---

### Task 1: Freeze E0/E1 AMP policy and formal config validation

**Files:**
- Modify: `tests/test_train_cli.py`
- Modify: `tests/test_model_configs.py`
- Modify: `configs/models/tsvit_baseline.yaml`
- Modify: `configs/models/tsvit_wpe_basic.yaml`
- Modify: `scripts/train_e0.py`

**Interfaces:**
- Consumes: a formal E0 or E1 YAML document.
- Produces: `load_formal_config(path) -> dict[str, Any]` and `formal_output_for(config) -> Path` with strict experiment/WPE pairing.

- [x] **Step 1: Write failing config-policy tests**

Add assertions to `tests/test_train_cli.py`:

```python
def test_formal_e1_config_is_accepted_with_frozen_amp_retry() -> None:
    config = load_formal_config(ROOT / "configs/models/tsvit_wpe_basic.yaml")
    assert config["experiment"]["id"] == "E1"
    assert config["model"]["wavelet"]["enabled"] is True
    assert config["training"]["amp_backoff_factor"] == 0.5
    assert config["training"]["amp_min_scale"] == 128.0
    assert config["training"]["amp_max_backoffs_per_batch"] == 6
    assert formal_output_for(config).name == "e1_tsvit_doy_wpe_seed42"


def test_formal_e1_rejects_disabled_wpe(tmp_path: Path) -> None:
    config = yaml.safe_load((ROOT / "configs/models/tsvit_wpe_basic.yaml").read_text())
    config["model"]["wavelet"]["enabled"] = False
    path = tmp_path / "bad-e1.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="E1 requires WPE enabled"):
        load_formal_config(path)
```

Extend the frozen training dictionary expectation with the three new fields. In `tests/test_model_configs.py`, compare `data`, `training`, and `evaluation` for E0/E1 equality in addition to the existing non-WPE model comparison.

- [x] **Step 2: Run tests and verify RED**

Run:

```powershell
D:\Anaconda3\envs\cawa\python.exe -m pytest tests\test_train_cli.py tests\test_model_configs.py -q
```

Expected: FAIL because E1 is rejected, `formal_output_for` is missing, and retry fields are absent.

- [x] **Step 3: Add frozen fields to both YAML files**

Insert under `training` in both configs:

```yaml
amp_backoff_factor: 0.5
amp_min_scale: 128.0
amp_max_backoffs_per_batch: 6
```

- [x] **Step 4: Generalize strict formal config validation**

In `scripts/train_e0.py`, replace the single output constant and E0-only checks with:

```python
FORMAL_OUTPUTS = {
    "E0": Path(r"E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e0_tsvit_doy_seed42"),
    "E1": Path(r"E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e1_tsvit_doy_wpe_seed42"),
}


def formal_output_for(config: Mapping[str, Any]) -> Path:
    experiment_id = str(config["experiment"]["id"])
    try:
        return FORMAL_OUTPUTS[experiment_id]
    except KeyError as error:
        raise ValueError(f"unsupported formal experiment {experiment_id}") from error
```

Add `Mapping` to the existing `typing` import. Permit only `E0` and `E1`; require disabled WPE for E0 and enabled WPE for E1. For E1, compare its wavelet mapping exactly with the approved values. Add the three retry keys to `expected_training`. Change `--output-dir` default to `None`, then choose `args.output_dir or formal_output_for(config)` after loading the config.

- [x] **Step 5: Verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [x] **Step 6: Review checkpoint without Git mutation**

Run `git diff --check` and inspect only the five Task 1 files. Do not stage or commit.

---

### Task 2: Implement auditable same-group AMP retry with TDD

**Files:**
- Create: `tests/test_amp_same_batch_retry.py`
- Modify: `src/training/trainer.py`
- Modify: `tests/test_amp_scaler_policy.py`

**Interfaces:**
- Consumes: one already-augmented effective group plus sample metadata.
- Produces: one successful optimizer transition and group loss, or a diagnosed `FloatingPointError`; appends JSON objects to `amp_events.jsonl`.

- [x] **Step 1: Write a CUDA test for transient overflow and exact state advancement**

Create a one-group dataset/model fixture using `window_size=4`, physical/effective batch 4 and a gradient hook that returns `Inf` only on its first call. Track model forward calls and optimizer `step` calls. Assert:

```python
assert metrics["optimizer_steps"] == 1
assert trainer.state.global_step == 1
assert optimizer_step_calls == 1
assert model.forward_calls == 2
assert trainer.scaler.get_scale() == 4096.0
events = [json.loads(line) for line in (tmp_path / "amp_events.jsonl").read_text().splitlines()]
assert [event["event"] for event in events] == ["amp_gradient_backoff"]
assert events[0]["old_scale"] == 8192.0
assert events[0]["new_scale"] == 4096.0
assert events[0]["attempt"] == 1
```

Mark the test `skipif(not torch.cuda.is_available())` because it validates real CUDA GradScaler state.

- [x] **Step 2: Write a CUDA test for terminal minimum-scale failure**

Use `amp_init_scale=256`, `amp_min_scale=128`, `amp_max_backoffs_per_batch=1` and an always-Inf gradient hook. Assert `FloatingPointError`, unchanged model parameters, zero optimizer calls, unchanged scheduler step, `global_step == 0`, and two audit events ending in `amp_gradient_failure` at scale 128.

- [x] **Step 3: Run tests and verify RED**

Run:

```powershell
D:\Anaconda3\envs\cawa\python.exe -m pytest tests\test_amp_same_batch_retry.py -q
```

Expected: FAIL because `TrainerConfig` has no backoff fields and the trainer terminates on the first non-finite gradient.

- [x] **Step 4: Add validated retry configuration**

Add fields to `TrainerConfig`:

```python
amp_backoff_factor: float = 0.5
amp_min_scale: float = 128.0
amp_max_backoffs_per_batch: int = 6
```

Validate `0 < factor < 1`, `0 < min_scale <= init_scale`, and non-negative integer backoffs. Extend `tests/test_amp_scaler_policy.py` to assert all frozen values.

- [x] **Step 5: Extract gradient diagnostics and audit writer**

Add focused methods:

```python
def _gradient_diagnostics(self) -> list[dict[str, Any]]: ...

def _append_amp_event(self, event: Mapping[str, Any]) -> None:
    self.output_dir.mkdir(parents=True, exist_ok=True)
    with (self.output_dir / "amp_events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True) + "\n")
        stream.flush()
```

The diagnostic payload must retain parameter name, NaN count, Inf count, and finite absolute maximum.

- [x] **Step 6: Wrap the complete effective group in a retry loop**

Move zeroing, all microbatch forward/backward work, unscale, diagnostics and step inside `while True`. Keep augmentation outside. On failure compute:

```python
old_scale = float(self.scaler.get_scale())
can_backoff = (
    backoffs < self.config.amp_max_backoffs_per_batch
    and old_scale > self.config.amp_min_scale
)
new_scale = max(
    self.config.amp_min_scale,
    old_scale * self.config.amp_backoff_factor,
)
event_name = "amp_gradient_backoff" if can_backoff else "amp_gradient_failure"
```

Write the full event before state changes. If retrying, call `optimizer.zero_grad(set_to_none=True)` and `scaler.update(new_scale)`, increment `backoffs`, and continue. If terminal, clear gradients, finalize scaler state without reducing below 128, write `nonfinite_diagnostic.json`, and raise. Only the finite branch may call `scaler.step`, normal `scaler.update`, `scheduler.step`, and increment counters.

- [x] **Step 7: Verify GREEN and regression behavior**

Run:

```powershell
D:\Anaconda3\envs\cawa\python.exe -m pytest tests\test_amp_same_batch_retry.py tests\test_amp_scaler_policy.py tests\test_trainer.py tests\test_checkpoint_resume.py tests\test_checkpoint_cuda_resume.py -q
```

Expected: all tests pass. Confirm the transient test reports exactly one optimizer step and the terminal test reports none.

- [x] **Step 8: Review checkpoint without Git mutation**

Run `git diff --check`; inspect the new test and trainer diff. Do not stage or commit.

---

### Task 3: Make the real-data CUDA smoke exercise E1 policy

**Files:**
- Modify: `scripts/smoke_e0.py`
- Modify: `tests/test_script_entrypoints.py`

**Interfaces:**
- Consumes: `--config configs/models/tsvit_wpe_basic.yaml` and the existing Train/Validation-only builder.
- Produces: `.smoke/e1-amp-retry/smoke_result.json`, checkpoint round-trip evidence, peak memory and zero Test records loaded.

- [x] **Step 1: Write a failing entrypoint/config test**

Add a subprocess `--help` test and a unit assertion that the smoke parser accepts an E1 config/output directory. Expected RED: the current parser cannot be passed an argv and the output metadata is E0-specific.

- [x] **Step 2: Generalize smoke labels and scaler policy**

Rename user-facing strings from E0 to formal E0/E1 without renaming the file. Initialize GradScaler with the config values and add the retry fields, experiment ID, parameter count, `torch.cuda.max_memory_allocated()`, selected physical/effective batch, and `test_records_loaded: 0` to `smoke_result.json`. Preserve real Train sample, one Validation tile, finite overfit, and exact checkpoint reload checks.

- [x] **Step 3: Run script tests**

Run:

```powershell
D:\Anaconda3\envs\cawa\python.exe -m pytest tests\test_script_entrypoints.py tests\test_train_cli.py -q
```

Expected: all tests pass.

- [x] **Step 4: Run full static and unit verification**

Run:

```powershell
D:\Anaconda3\envs\cawa\python.exe -m pytest -q --basetemp=.smoke\pytest-e1-prelaunch
D:\Anaconda3\envs\cawa\python.exe -m compileall -q src scripts tests
git diff --check
```

Expected: all tests pass, compileall exit 0, diff check empty.

- [x] **Step 5: Run the real E1 CUDA smoke**

Run:

```powershell
D:\Anaconda3\envs\cawa\python.exe scripts\smoke_e0.py --config configs\models\tsvit_wpe_basic.yaml --output-dir .smoke\e1-amp-retry --overfit-steps 8
```

Expected: status passed, physical batch 16 selected, finite decreasing losses, exact checkpoint reload, finite single-tile Validation, peak memory recorded, and Test records loaded equals 0.

---

### Task 4: Freeze records and launch formal E1

**Files:**
- Modify: `docs/DECISIONS.md`
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/STATUS.md`
- Modify: `docs/TASKS.md`
- Create: `docs/handoffs/2026-07-18-e1-formal-training-started.md`
- Create outside Git: `E:\maize_paper_workspace\00_docs\experiment_notes\maize_model_research_e1_tsvit_doy_wpe_seed42_2026-07-18.md`
- Modify outside Git: `E:\maize_paper_workspace\00_docs\inventories\agent_project_handoff_2026-06-11.md`

**Interfaces:**
- Consumes: verified source/config hashes, smoke result and E1 command.
- Produces: auditable experiment-start records and a hidden formal training process.

- [x] **Step 1: Record the frozen retry decision and E1 launch configuration**

Add a new decision after D-019 with the exact four AMP fields and no-skip semantics. Register E1 as `running`, including branch/HEAD/dirty status, manifest and normalization hashes, seed, model/WPE parameters, output path, command, smoke result and the statement that Test was not loaded.

- [x] **Step 2: Preflight the formal output directory**

Confirm no `train_e0.py` process is running and the E1 output directory does not contain a prior formal `last.pt`. Do not overwrite or resume an unrecognized run. Create the new output directory only after this check.

- [x] **Step 3: Launch formal E1 in a hidden process**

Launch from the repository with:

```powershell
D:\Anaconda3\envs\cawa\python.exe scripts\train_e0.py --config configs\models\tsvit_wpe_basic.yaml --output-dir E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e1_tsvit_doy_wpe_seed42 --physical-batch-size 16
```

Redirect stdout/stderr to the output directory and record PID. Do not pass `--resume` for the fresh run.

- [x] **Step 4: Verify launch health**

Check `run_snapshot.json`: experiment E1, WPE enabled, seed 42, physical/effective batch 16/16, retry policy exact, manifest/normalization hashes exact, Test absent. Verify the process is alive, stderr empty, GPU memory/utilization plausible, and the first optimizer steps advance without an unauthorized skip.

- [x] **Step 5: Refresh formal workspace inventory**

After experiment-start records exist, run the mandated inventory refresh and verify the E1 note/output entries appear in `full_file_manifest_2026-06-11.csv`.

- [x] **Step 6: Set monitoring boundary**

Monitor `metrics.jsonl`, `amp_events.jsonl`, stderr and PID. If a backoff occurs, verify the same batch succeeds and state advances once. If terminal failure occurs at scale 128, preserve evidence and debug before resuming. At early stopping or epoch 100, independently replay full Validation from `best.pt`; do not evaluate Test.

- [x] **Step 7: Final review without publication**

Report PID, output path, exact E1 configuration, parameter count, smoke peak VRAM, initial health and monitoring state. Do not stage, commit, push or create a PR.
