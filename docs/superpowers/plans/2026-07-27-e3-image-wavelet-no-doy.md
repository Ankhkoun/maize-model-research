# E3 Image-Wavelet No-DOY Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a controlled TSViT ablation that uses image content plus K5 Mexican-hat wavelet encoding without calendar DOY or learned temporal slots.

**Architecture:** Add the `none` temporal-position route so K5 WPE consumes only patch tokens and the valid-frame mask. Register E3 with the same data, optimisation and validation-selection controls as E0/E2-WPEonly; do not create or access a Test asset.

**Tech Stack:** Python 3.11, PyTorch, PyYAML, pytest.

## Global Constraints

- E3 must use Xinjiang 2021 Train 495 / Validation 276, seed 42, 24x24 windows and patch size 2.
- Model selection is by Validation maize IoU only; Test must not be loaded, run or registered.
- E3 must use K5 content-only Mexican-hat WPE and `temporal_position_encoding.kind: none`.

---

### Task 1: Add strict no-DOY temporal routing

**Files:**
- Modify: `src/models/tsvit_segmentation.py`
- Test: `tests/test_e3_image_wavelet_no_doy.py`

**Interfaces:**
- Consumes: `TSViTSegmentation.forward(images, doy, valid_mask)`.
- Produces: `temporal_position_encoding.kind="none"`, with logits invariant to valid calendar-DOY values for fixed imagery.

- [ ] Write a test with fixed images and two valid DOY calendars; require bitwise-equal logits for the `none` route.
- [ ] Run `python -m pytest -q tests/test_e3_image_wavelet_no_doy.py` and verify it fails because `none` is unsupported.
- [ ] Permit `none`; do not construct a DOY embedding for it; route K5 WPE from patch tokens and the valid-frame mask only.
- [ ] Re-run the targeted test and require it to pass.

### Task 2: Freeze the E3 formal experiment

**Files:**
- Create: `configs/models/tsvit_e3_image_wavelet_no_doy_mexican_hat_k5.yaml`
- Modify: `scripts/train_e0.py`
- Test: `tests/test_e3_image_wavelet_no_doy.py`

**Interfaces:**
- Consumes: `load_formal_config(path)` and `formal_output_for(config)`.
- Produces: formal ID `E3-ImageWPE` with output directory `e3_tsvit_image_five_point_mexican_hat_k5_seed42`.

- [ ] Write tests for the E3 configuration and for rejection of `doy_lookup`.
- [ ] Run the targeted test and verify it fails because the configuration is absent.
- [ ] Add the configuration and exact formal-configuration checks for no temporal position encoding and frozen K5 parameters.
- [ ] Re-run the targeted test and require it to pass.

### Task 3: Verify and launch Train/Validation

**Files:**
- Modify: `docs/DECISIONS.md`, `docs/EXPERIMENTS.md`, `docs/STATUS.md`, `docs/TASKS.md`
- Create: `docs/handoffs/2026-07-27-e3-image-wavelet-no-doy-*.md`

**Interfaces:**
- Consumes: the frozen E3 configuration, manifest and train-only normalisation.
- Produces: a CUDA smoke record, formal Train/Validation artifacts and an independent Validation replay; no Test artifact.

- [ ] Run a one-epoch CUDA smoke with a separate smoke output directory and confirm Train/Validation-only loading.
- [ ] Start formal E3 Train/Validation with physical/effective batch 16/16 and retain logs, checkpoints and AMP events in the formal output directory.
- [ ] In a new process, replay the frozen best checkpoint on all 276 Validation tiles.
- [ ] Update the experiment records with hashes and metrics, then run full pytest, compileall and `git diff --check`.
