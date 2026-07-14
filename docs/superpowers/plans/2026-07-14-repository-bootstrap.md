# Maize Model Research Repository Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a private, code-first repository for the maize model research while keeping large data and outputs in the existing E-drive workspace.

**Architecture:** The Git repository contains project rules, relative-path configuration, experiment metadata, manifests and future source code. A Git-ignored local configuration points to `E:\maize_paper_workspace`; external research implementations remain isolated references.

**Tech Stack:** Git, GitHub private repository, YAML configuration, Markdown research records, Python/PyTorch in later phases.

## Global Constraints

- Do not move or modify the formal data workspace during repository bootstrap.
- Do not commit local absolute paths, data, pseudo-labels, checkpoints or generated outputs.
- Do not delete any existing handoff or workspace file during consolidation.

---

### Task 1: Establish repository documentation and boundaries

**Files:**
- Create: `.gitignore`, `README.md`, `AGENTS.md`
- Create: `docs/STATUS.md`, `docs/TASKS.md`, `docs/DECISIONS.md`
- Create: `docs/DATA_CONTRACT.md`, `docs/EXPERIMENTS.md`, `docs/PATHS.md`
- Create: `docs/research/EXACT_WPE_SPECIFICATION.md`
- Create: `docs/WORKSPACE_REVIEW.md`

- [x] Consolidate verified facts from the three existing handoff directories.
- [x] State the data/Git boundary and external-tree safety constraints.
- [x] Record completed PEACE-Net results and planned E0/E1 experiments.

### Task 2: Establish portable local path configuration

**Files:**
- Create: `configs/paths.example.yaml`
- Create but ignore: `configs/paths.local.yaml`

- [x] Commit only a portable placeholder path.
- [x] Configure the local workspace root as `E:/maize_paper_workspace`.
- [ ] Verify `paths.local.yaml` is ignored by Git.

### Task 3: Initialize and publish the private repository

- [ ] Initialize Git with branch `main`.
- [ ] Verify no ignored data or local path file is staged.
- [ ] Create the first commit.
- [ ] Create the GitHub repository `maize-model-research` as private.
- [ ] Push `main` and verify remote visibility.
