# E2-W P_T + Five-Point Mexican-Hat WPE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and train E2-W, a TSViT variant using a learned 26-slot temporal position table and a content-only learnable five-point Mexican-hat residual WPE.

**Architecture:** Preserve the frozen E0/E1 DOY lookup and legacy WPE classes. Add a new E2-W-only local wavelet module that normalizes content tokens, applies three shared Mexican-hat kernels over offsets `[-2,-1,0,1,2]`, mean-centers/L1-normalizes every kernel, gates each channel and adds a small residual. E2-W uses a learned `[1,26,D]` P_T table and never sends it through the wavelet operator.

**Tech Stack:** Python 3.11, PyTorch, pytest, YAML, CUDA via `D:\Anaconda3\envs\cawa\python.exe`.

## Global Constraints

- E2-W is Validation-selected only; do not load Test records or run Test during training.
- Xinjiang 2021, 24x24, patch2, seed42, Train495/Validation276, pseudo-label supervision, normalization and training schedule remain fixed.
- WPE: `kernel_size=5`, `delta_t=[-2,-1,0,1,2]`, scale init `[0.75,1.00,1.25]`, bounds `[0.50,1.50]`, shift init `[0,0,0]`, bounds `[-1,1]`.
- Keep E0/E1 checkpoint loading and outputs compatible. Do not modify frozen assets.
- No Test-driven choices, checkpoint selection or threshold search.

### Task 1: Add failing unit tests for E2-W wavelet invariants

**Files:**
- Modify: `tests/test_wavelet_position_encoding.py`
- Modify: `tests/test_tsvit_wpe_equivalence.py`

- [ ] **Step 1: Write failing tests** for a new `FivePointMexicanHatWaveletEncoding` that assert bounded learned `scales`/`shifts`, each generated kernel has five coefficients, sum exactly zero within numerical tolerance, L1 norm one, invalid frames do not affect valid responses, and finite backward gradients.
- [ ] **Step 2: Run RED**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_wavelet_position_encoding.py --basetemp=.smoke\pytest-e2w-wavelet-red`

Expected: import failure because the E2-W module does not exist.

- [ ] **Step 3: Add failing model tests** asserting E2-W uses `temporal_slot_embedding` `[1,T,D]`, sends raw patch content—not `P_T`—to WPE, and `gamma=0` makes the enabled E2-W path exactly equal to the same P_T model with WPE disabled.
- [ ] **Step 4: Run RED**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_tsvit_wpe_equivalence.py --basetemp=.smoke\pytest-e2w-model-red`

Expected: failure for the missing E2-W configuration/API.

### Task 2: Implement the content-only five-point Mexican-hat operator

**Files:**
- Modify: `src/models/wavelet_position_encoding.py`

- [ ] **Step 1: Implement `FivePointMexicanHatWaveletEncoding(dim, scales_init, scale_bounds, shifts_init, shift_bounds, gamma_init, eps)`.** Its `kernels` property computes `u=(delta_t-b)/sigma`, `psi=(1-u^2)*exp(-u^2/2)`, then `(psi-psi.mean()) / psi.abs().sum().clamp_min(eps)` for three bases.
- [ ] **Step 2: Implement forward** as `content_norm=LayerNorm(tokens)`; use replicate-index local gathering for offsets -2..2, apply each shared kernel to every channel, gate with `sigmoid(raw_gates[R,D])`, sum bases, output-normalize, mask invalid queries, and return `tokens + gamma * residual`.
- [ ] **Step 3: Run GREEN**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_wavelet_position_encoding.py --basetemp=.smoke\pytest-e2w-wavelet-green`

Expected: PASS.

### Task 3: Add learned P_T and E2-W routing without changing E0/E1

**Files:**
- Modify: `src/models/tsvit_segmentation.py`
- Modify: `tests/test_tsvit_wpe_equivalence.py`

- [ ] **Step 1: Add `temporal_position_encoding.kind` with legacy default `doy_lookup` and E2-W `learned_slot`.** The new path owns `temporal_slot_embedding = nn.Parameter(torch.randn(1,num_frames,dim))`; the legacy path keeps `doy_embedding` unchanged.
- [ ] **Step 2: Route E2-W as** `wavelet_content = five_point_wavelet(patch_tokens, valid_mask)` then `temporal_tokens = patch_tokens + P_T + (wavelet_content - patch_tokens)`. Legacy E1 remains `legacy_wavelet(patch_tokens + doy_tokens, doy, valid_mask)`.
- [ ] **Step 3: Run GREEN**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_tsvit_wpe_equivalence.py --basetemp=.smoke\pytest-e2w-model-green`

Expected: PASS, including exact gamma-zero equivalence.

### Task 4: Freeze E2-W config and training guard

**Files:**
- Create: `configs/models/tsvit_e2w_pt_mexican_hat_k5.yaml`
- Modify: `scripts/train_e0.py`
- Modify: `tests/test_train_cli.py`

- [ ] **Step 1: Write failing config tests** requiring experiment id `E2W`, P_T kind `learned_slot`, the exact five-point wavelet dictionary, the unchanged training dictionary, Validation-only evaluation, and formal output `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2w_tsvit_pt_mexican_hat_k5_seed42`.
- [ ] **Step 2: Run RED**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_train_cli.py --basetemp=.smoke\pytest-e2w-config-red`

Expected: E2W is rejected before implementation.

- [ ] **Step 3: Add the YAML and exact `E2W` guard/output mapping** while leaving E0/E1 checks unchanged.
- [ ] **Step 4: Run GREEN**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests/test_train_cli.py --basetemp=.smoke\pytest-e2w-config-green`

Expected: PASS.

### Task 5: Verify, document and launch the Validation-only run

**Files:**
- Modify: `docs/DECISIONS.md`
- Modify: `docs/EXPERIMENTS.md`
- Modify: `docs/STATUS.md`
- Modify: `docs/TASKS.md`
- Create: `docs/handoffs/2026-07-19-e2w-launch.md`
- Create: `E:\maize_paper_workspace\00_docs\experiment_notes\maize_model_research_e2w_pt_mexican_hat_k5_seed42_2026-07-19.md`

- [ ] **Step 1: Run full verification.**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q --basetemp=.smoke\pytest-e2w-full`

Run: `D:\Anaconda3\envs\cawa\python.exe -m compileall -q src scripts tests`

Run: `git diff --check`

- [ ] **Step 2: Run one minimal CUDA smoke** using only Train/Validation records and the E2-W config; verify finite logits/loss/gradients and that Test records loaded is zero.
- [ ] **Step 3: Register the frozen E2-W hypothesis/configuration before formal training.** Record all continuous display metrics to five decimals.
- [ ] **Step 4: Launch exactly one Train/Validation run** with the existing physical/effective batch16/16 and output directory. Do not launch Test evaluation.
- [ ] **Step 5: After training, replay the frozen best Validation checkpoint, record learned scales/shifts/gamma/gates, update research and workspace handoffs, refresh workspace inventory, and run final verification.**
