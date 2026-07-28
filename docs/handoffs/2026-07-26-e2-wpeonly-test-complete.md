# E2-WPEonly frozen Test complete

- Date: 2026-07-26
- Root: `D:\cj_swcc\maize-model-research`
- Branch/HEAD: `codex/e2-wpeonly` / `dfbff1bf91c4e80e7077f2686e6ecd155b20ae2e`

## Verified session work

The user explicitly authorized final Test evaluation for E2-WPEonly. Added a frozen Test registry entry in `scripts/evaluate_test.py` and two focused TDD assertions. The initial red test failed with the expected missing-key error; after registration, the focused suite passed (12 passed). The complete gate then passed: `104 passed`, `compileall`, and `git diff --check`.

One pseudo-label Test process and one independent ground-truth Test process were started after the output directories were confirmed absent. Both script invocations outlived the desktop shell timeout but were verified as continuing processes and each completed exactly once; do not rerun because their output directories are now one-time artifacts.

## Frozen asset and results

- checkpoint: `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e2_wpeonly_tsvit_doy_mexican_hat_k5_seed42\best.pt`
- SHA256: `1C5F697D5FF12D114295D1FD359ADE1AA90342DE71DEAA3D5A3A1BD69130EBD8`
- epoch/global step/Validation IoU: `11` / `36597` / `0.9375021026721523`
- pseudo Test IoU/F1/Kappa: `0.9263511346358012` / `0.9617676839699721` / `0.894868750435993`
- native30m Test IoU/F1/Kappa: `0.7801353730286065` / `0.8764899398648934` / `0.7773863586240127`
- upsampled10m Test IoU/F1/Kappa: `0.771874547760831` / `0.8712519164929268` / `0.7684488674917034`
- each output reports 305 evaluated tiles and an exactly conserved confusion-matrix total.

E0 remains best under the preregistered Validation maize-IoU selection rule. E2-WPEonly does not exceed E0 on reported Test IoU/F1/Kappa; Test did not change model selection, checkpoint, threshold, code architecture, or training.

## Session-owned files

- `scripts/evaluate_test.py`
- `tests/test_test_evaluation_portability.py`
- `tests/test_e2_wpeonly.py`
- `docs/EXPERIMENTS.md`
- `docs/STATUS.md`
- `docs/TASKS.md`
- `docs/DECISIONS.md`
- this handoff

Pre-existing session documentation edits in those four docs and `docs/handoffs/2026-07-23-e2-wpeonly-validation-complete.md` were preserved; do not assume they are separately attributable from this session without review.

## Next task

User review before any stage, commit, or push. No further Test evaluation is permitted for E2-WPEonly because both one-time output directories now exist.