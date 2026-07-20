# E0/E1 Ground-Truth Test 双口径实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为冻结 E0/E1 以一次模型拼接同时生成原生 30m 与标签复制到模型网格的真实标签 Test 指标。

**Architecture:** 新建小型 ground-truth evaluator，复用现有 Test-only 数据集、完整滑窗 logits 拼接和 `ConfusionMatrix`。它按 sample ID 加载 85x85 独立标签，在内存中由同一份 256x256 logits 衍生两个尺度的概率、loss 与指标；CLI 只加载固定 checkpoint 并拒绝重复输出。

**Tech Stack:** Python 3.11、PyTorch、NumPy、现有 manifest/dataset/window/metrics 模块。

## Global Constraints

- 仅 Test 305；真实标签根目录相对于正式工作区为 `03_processed_data/labels_30m/xinjiang_2021/2021`。
- 仅 `label != 255`；不使用耕地掩膜、排除列表或 Test 驱动阈值搜索。
- 固定 0.5、E0 epoch13/E1 epoch8、现有哈希与完整 24x24 滑窗拼接。
- 不 stage、commit、push 或创建 PR。

---

### Task 1: 双尺度标签与指标内核

**Files:**
- Create: `src/training/ground_truth_evaluate.py`
- Test: `tests/test_ground_truth_evaluate.py`

**Interfaces:**
- Produces: `evaluate_ground_truth_tiles(model, dataset, ground_truth_root, device, window_size, stride, window_batch_size, amp) -> dict[str, dict]`.
- Produces metrics keys `native30m` and `upsampled10m`, each compatible with the current metric document format.

- [ ] **Step 1: Write failing unit tests** for `(85,85)` labels, invalid values/shapes, 3x3 probability averaging, 3x3 label replication, 255 border exclusion, confusion conservation, and rejection of non-Test samples.
- [ ] **Step 2: Run the focused tests** and confirm the evaluator import/function is absent.
- [ ] **Step 3: Implement the minimal evaluator**: stitch logits once; load `<ground_truth_root>/<sample_id>/y_patch_30m.npy`; average softmax probabilities over `3x3` for native30m; repeat labels to `255x255` for upsampled10m; calculate NLL and update independent confusion matrices.
- [ ] **Step 4: Run focused tests** and confirm both metric dictionaries pass.
- [ ] **Step 5: Do not commit**; Git publication remains user-controlled.

### Task 2: Frozen CLI, provenance, and TDD guards

**Files:**
- Create: `scripts/evaluate_test_ground_truth.py`
- Test: `tests/test_test_ground_truth_cli.py`
- Modify: `scripts/evaluate_test.py`

**Interfaces:**
- Consumes: `FROZEN_TEST_ASSETS`, `build_test_dataset`, frozen checkpoint loader, and the Task 1 evaluator.
- Produces: `test_evaluation_ground_truth/run_snapshot.json`, `native30m/test_evaluation.json`, and `upsampled10m/test_evaluation.json` below each frozen checkpoint parent.

- [ ] **Step 1: Write failing CLI tests** for mandatory config/checkpoint, relative ground-truth root resolution, fixed output paths, 305 tiles, two documents, and one-time output reservation.
- [ ] **Step 2: Run them** and confirm failure before the entry point exists.
- [ ] **Step 3: Implement the CLI** with CUDA-only execution, frozen asset/hash/config guards, relative formal label root, atomic JSON writes, and a shared snapshot.
- [ ] **Step 4: Repair the frozen training metadata check** to compare the stable frozen fields (including effective batch 16 and early stopping 12) while accepting E0's legitimate absence of later E1 AMP-retry fields.
- [ ] **Step 5: Run focused tests** and the actual E0/E1 metadata preflight without loading Test samples.

### Task 3: Formal evaluation and records

**Files:**
- Modify: `docs/EXPERIMENTS.md`, `docs/STATUS.md`, `docs/TASKS.md`, `docs/DECISIONS.md`
- Create: `docs/handoffs/2026-07-19-e0-e1-ground-truth-test-evaluation-complete.md`

- [ ] **Step 1: Run fresh pytest and a minimal CUDA smoke** before formal evaluation.
- [ ] **Step 2: In separate new processes, run E0 then E1 exactly once** into their absent ground-truth output directories.
- [ ] **Step 3: Audit** `evaluated_tiles=305`, both supervised-pixel counts, full matrices, metric deltas, checkpoint/output SHA256, and the two-scale interpretation.
- [ ] **Step 4: Update repository and formal-workspace records**, preserving pseudo-label results as diagnostics and identifying ground-truth metrics as the final held-out reference evaluation.
- [ ] **Step 5: Run fresh full pytest, compileall, and `git diff --check`; do not commit.**
