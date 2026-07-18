# E0 Formal Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and run the approved Xinjiang 2021 E0 TSViT formal training pipeline with deterministic 24x24 windows and validation-only checkpoint selection.

**Architecture:** A repository manifest resolves complete parent tiles under `workspace_root`. The loader reads one contiguous 256x256 cube at a time, creates deterministic 24x24 windows in memory, and feeds window mini-batches to a shared TSViT model. A focused trainer owns normalization, AMP, scheduling, metrics, checkpointing, resume, validation stitching, and early stopping.

**Tech Stack:** Python 3.11, PyTorch 2.10, NumPy, PyYAML, pytest, CUDA/AMP.

## Global Constraints

- E0 uses `image_size=24`, `patch_size=2`, 26 frames, 10 bands, 2 classes, seed 42.
- DOY uses a learned table with index 0 reserved for padding and indices 1..366 for calendar days.
- E0 and E1 must share every non-WPE setting; this plan runs E0 only.
- Loss is ordinary cross entropy with `ignore_index=255`; `pseudo_confidence` is excluded.
- Train/Validation/Test parent-tile split is 495/276/305 and must never be randomized across regions.
- Band statistics use Train only. Model/checkpoint selection uses Validation only. Test is not loaded.
- Formal outputs go to `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e0_tsvit_doy_seed42`.
- Preserve the existing dirty worktree. Do not stage, commit, push, reset, clean, or modify `_external` trees.

---

### Task 1: Formal E0 model configuration and learned DOY table

**Files:**
- Modify: `configs/models/tsvit_baseline.yaml`
- Modify: `configs/models/tsvit_wpe_basic.yaml`
- Modify: `src/models/tsvit_segmentation.py`
- Modify: `tests/test_model_configs.py`
- Create: `tests/test_learned_doy_embedding.py`

**Interfaces:**
- Consumes: model configuration mapping and tensors `images [B,T,C,24,24]`, `doy [B,T]`, `valid_mask [B,T]`.
- Produces: `TSViTSegmentation` with `nn.Embedding(367, dim, padding_idx=0)` and logits `[B,2,24,24]`.

- [ ] **Step 1: Write failing configuration and DOY tests**

```python
def test_formal_configs_use_24px_windows_and_2px_patches():
    assert e0["model"]["image_size"] == 24
    assert e0["model"]["patch_size"] == 2
    assert e1["model"]["image_size"] == 24
    assert e1["model"]["patch_size"] == 2

def test_doy_zero_is_padding_and_calendar_days_are_learned():
    model = TSViTSegmentation(config)
    assert isinstance(model.doy_embedding, torch.nn.Embedding)
    assert model.doy_embedding.num_embeddings == 367
    assert model.doy_embedding.padding_idx == 0
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_model_configs.py tests/test_learned_doy_embedding.py`

Expected: failures showing `256/8` and `nn.Linear`.

- [ ] **Step 3: Implement the minimal formal configuration and lookup table**

```python
self.doy_embedding = nn.Embedding(367, self.dim, padding_idx=0)
doy_indices = torch.where(valid_mask, doy.round().long(), torch.zeros_like(doy, dtype=torch.long))
if valid_mask.any() and ((doy_indices[valid_mask] < 1) | (doy_indices[valid_mask] > 366)).any():
    raise ValueError("valid DOY values must be integer calendar days in [1,366]")
doy_tokens = self.doy_embedding(doy_indices)
```

- [ ] **Step 4: Run focused and existing model tests and verify GREEN**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_model_configs.py tests/test_learned_doy_embedding.py tests/test_tsvit_wpe_equivalence.py tests/test_e1_robustness.py`

Expected: all selected tests pass.

---

### Task 2: Repository manifest and full-tile data contract

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/manifest.py`
- Create: `src/data/xinjiang_tiles.py`
- Create: `scripts/build_xinjiang_2021_manifest.py`
- Create: `manifests/xinjiang_2021_e0_e1.csv`
- Create: `tests/test_manifest.py`
- Create: `tests/test_xinjiang_tiles.py`

**Interfaces:**
- Produces: `TileRecord(sample_id, region_id, split, cube_path, label_path, metadata_path, time_quality_path)`.
- Produces: `XinjiangTileDataset(records, workspace_root, normalization=None)` returning images, label, DOY, valid mask, sample ID, and region ID.

- [ ] **Step 1: Write failing manifest and loader tests using temporary sample directories**

```python
def test_build_records_resolves_relative_workspace_paths(tmp_path):
    records = build_records(source_rows, tmp_path)
    assert records[0].cube_path.as_posix().endswith("sample_a/x_cube.npy")
    assert records[0].label_path.as_posix().endswith("sample_a/pseudo_label.npy")

def test_tile_loader_uses_slot_midpoint_doy_and_all_resolved_slots(tmp_path):
    sample = dataset[0]
    assert sample["images"].shape == (26, 10, 256, 256)
    assert sample["doy"].dtype == torch.float32
    assert sample["valid_mask"].all()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_manifest.py tests/test_xinjiang_tiles.py`

Expected: import failures because the data package does not exist.

- [ ] **Step 3: Implement strict records, metadata validation, midpoint DOY, and split checks**

The loader must require `[26,10,256,256] float32`, bands `B2..B12` in the approved order, labels in `{0,1,255}`, 26 ordered time-quality entries, and no overlap among parent sample IDs across splits. A time slot is valid when `resolved_pixel_count > 0`; its DOY is the calendar midpoint between `start_date` and `end_date`.

- [ ] **Step 4: Generate the repository manifest and verify counts/hash**

Run: `D:\Anaconda3\envs\cawa\python.exe scripts/build_xinjiang_2021_manifest.py --paths configs/paths.local.yaml`

Expected: `train=495 validation=276 test=305 total=1076`, source SHA256 `59C2...2B`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_manifest.py tests/test_xinjiang_tiles.py`

Expected: all selected tests pass.

---

### Task 3: Train-only normalization, deterministic windows, augmentation, and stitching

**Files:**
- Create: `src/data/normalization.py`
- Create: `src/data/windows.py`
- Create: `scripts/compute_train_normalization.py`
- Create: `manifests/xinjiang_2021_train_normalization.json`
- Create: `tests/test_normalization.py`
- Create: `tests/test_windows.py`

**Interfaces:**
- Produces: `compute_band_stats(dataset) -> dict[str, list[float] | int]`.
- Produces: `window_starts(256,24,24) -> [0,24,...,216,232]`.
- Produces: `extract_windows(images, label, starts, skip_all_ignore)` and `LogitAccumulator`.

- [ ] **Step 1: Write failing deterministic unit tests**

```python
def test_window_starts_anchor_final_window():
    assert window_starts(256, 24, 24) == [0,24,48,72,96,120,144,168,192,216,232]

def test_stitching_averages_overlaps_without_holes():
    accumulator = LogitAccumulator(2, 256, 256)
    # add all 121 constant windows
    logits = accumulator.finalize()
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_normalization.py tests/test_windows.py`

Expected: import failures.

- [ ] **Step 3: Implement streaming float64 Train-only statistics and window utilities**

Statistics must ignore invalid temporal slots, use float64 sum/squared-sum, and serialize mean/std/count plus manifest/source hashes. Augmentation applies identical flip/rotation operations to images and labels using a supplied `torch.Generator`.

- [ ] **Step 4: Compute and freeze real Train statistics**

Run: `D:\Anaconda3\envs\cawa\python.exe scripts/compute_train_normalization.py --manifest manifests/xinjiang_2021_e0_e1.csv --paths configs/paths.local.yaml`

Expected: 10 finite means, 10 positive finite standard deviations, `split=train`, 495 parent tiles.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_normalization.py tests/test_windows.py`

Expected: all selected tests pass.

---

### Task 4: Segmentation metrics and validation-only full-tile evaluation

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/metrics.py`
- Create: `src/training/evaluate.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_full_tile_evaluation.py`

**Interfaces:**
- Produces: `ConfusionMatrix(num_classes=2, ignore_index=255)`.
- Produces: `evaluate_tiles(model, dataset, normalization, window_batch_size, device) -> dict`.

- [ ] **Step 1: Write failing exact-value metric and stitching tests**

```python
def test_binary_metrics_match_known_confusion_matrix():
    cm = ConfusionMatrix(2, 255)
    cm.update(prediction, target)
    metrics = cm.compute()
    assert metrics["maize_iou"] == pytest.approx(tp / (tp + fp + fn))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_metrics.py tests/test_full_tile_evaluation.py`

Expected: import failures.

- [ ] **Step 3: Implement metrics and deterministic full-tile validation**

Metrics include loss, maize IoU/F1/precision/recall, mIoU, macro-F1, Kappa, area ratio, confusion matrix, and per-region aggregation. Evaluation must use `model.eval()`, inference mode, no random augmentation, and must reject records outside Validation unless explicitly running a final frozen Test command.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_metrics.py tests/test_full_tile_evaluation.py`

Expected: all selected tests pass.

---

### Task 5: AMP trainer, optimizer schedule, checkpoint, resume, and early stopping

**Files:**
- Create: `src/training/checkpoint.py`
- Create: `src/training/schedule.py`
- Create: `src/training/trainer.py`
- Create: `tests/test_schedule.py`
- Create: `tests/test_checkpoint_resume.py`
- Create: `tests/test_trainer.py`

**Interfaces:**
- Produces: `WarmupCosineSchedule(optimizer, total_steps, warmup_steps, start_lr, base_lr, min_lr)`.
- Produces: `save_checkpoint`, `load_checkpoint`, and `E0Trainer.fit()`.

- [ ] **Step 1: Write failing schedule/checkpoint/trainer tests**

```python
def test_resume_restores_epoch_optimizer_scheduler_scaler_and_rng(tmp_path):
    save_checkpoint(path, state)
    restored = load_checkpoint(path, model, optimizer, scheduler, scaler)
    assert restored.epoch == state.epoch
    assert torch.equal(torch.get_rng_state(), state.torch_rng_state)

def test_early_stopping_activates_after_warmup_and_patience():
    stopper = EarlyStopping(warmup_epochs=10, patience=12)
    assert not stopper.should_stop(epoch=10, metric=0.5)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_schedule.py tests/test_checkpoint_resume.py tests/test_trainer.py`

Expected: import failures.

- [ ] **Step 3: Implement minimal trainer behavior**

Use AdamW with weight decay 0, effective batch 16, FP16 autocast/GradScaler on CUDA, finite loss/gradient checks, per-step schedule, `last.pt` every epoch, `best.pt` on strict Validation maize-IoU improvement, and patience 12 after epoch 10. A NaN/Inf error writes diagnostic sample/window identifiers then terminates.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_schedule.py tests/test_checkpoint_resume.py tests/test_trainer.py`

Expected: all selected tests pass.

---

### Task 6: CLI, real-data smoke gates, and formal E0 launch

**Files:**
- Modify: `configs/models/tsvit_baseline.yaml`
- Create: `scripts/train_e0.py`
- Create: `scripts/smoke_e0.py`
- Create: `tests/test_train_cli.py`

**Interfaces:**
- Produces commands supporting `--smoke`, `--resume`, `--max-epochs`, `--output-dir`, and validation-only operation.

- [ ] **Step 1: Write failing CLI tests**

The test parses the formal config and asserts no Test dataset is constructed during training.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_train_cli.py`

- [ ] **Step 3: Implement CLI and immutable run snapshot**

The run snapshot contains resolved config, git HEAD/dirty file list, manifest/source hashes, normalization hash, selected physical/accumulation batch sizes, environment, and exact command.

- [ ] **Step 4: Run the complete automated suite**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q`

Expected: zero failures.

- [ ] **Step 5: Run real-data contract and one-batch overfit smoke**

Run: `D:\Anaconda3\envs\cawa\python.exe scripts/smoke_e0.py --paths configs/paths.local.yaml --manifest manifests/xinjiang_2021_e0_e1.csv`

Expected: contract pass, decreasing fixed-batch loss, finite gradients, selected CUDA batch size, and checkpoint reload equivalence.

- [ ] **Step 6: Launch formal E0 and monitor to terminal condition**

Run: `D:\Anaconda3\envs\cawa\python.exe scripts/train_e0.py --paths configs/paths.local.yaml --manifest manifests/xinjiang_2021_e0_e1.csv --config configs/models/tsvit_baseline.yaml --output-dir E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e0_tsvit_doy_seed42 --resume`

Expected: epochs continue until 100 or approved early stop; Test remains unread.

---

### Task 7: Independent validation replay and experiment records

**Files:**
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/STATUS.md`
- Modify: `docs/TASKS.md`
- Modify: `docs/DECISIONS.md` only if implementation changes a frozen behavior
- Create: `docs/handoffs/2026-07-17-e0-formal-training.md`
- Modify outside repository per fixed workflow: formal workspace handoff, experiment note, and inventories

- [ ] **Step 1: Reload best checkpoint in a new process and replay Validation**

Run the validation-only CLI against `best.pt` and compare the recomputed metrics with the saved best-epoch metrics using exact confusion counts and numerical tolerance for averaged loss.

- [ ] **Step 2: Record complete experiment provenance and results**

Record config, git state, seed, source/manifest/normalization hashes, command, output path, best/stopped epoch, metrics, checkpoint hashes, exceptions, and conclusion. Clearly label metrics as pseudo-label consistency rather than independent ground-truth accuracy.

- [ ] **Step 3: Refresh formal workspace inventory**

Run: `D:\Anaconda3\envs\cawa\python.exe E:\maize_paper_workspace\01_code\spring_maize_paper_dataset\refresh_workspace_inventory.py`

Expected: exit code 0 and updated inventory summaries.

- [ ] **Step 4: Run final repository verification**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q`

Expected: zero failures.

