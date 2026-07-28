# TSViT E0–E5 summary before seed replication

- Root: `D:\cj_swcc\maize-model-research`; branch: `codex/e5-doy-k9-wavelet`.
- Objective completed in this session: E4 strict image-only K9 and E5 calendar-DOY+K9 were implemented, formally trained, independently replayed on Validation, and each evaluated once on pseudo-label and annual-reference Test.
- Verified leader by frozen Validation IoU: E4 `0.938988` (F1 `0.968534`, Kappa `0.926002`), best epoch 11, SHA256 `E30E912C7ECEA90FAEFCCCF8444849F7F36A1137084FDDC9F70EC0F1204ABE14`.
- Verified leader by annual native30m reference: E0 IoU/F1/Kappa `0.785097/0.879613/0.782022`. E5 is second at `0.784859/0.879463/0.778732`, a native IoU difference of `-0.000238` from E0.
- E5: best epoch 14, checkpoint SHA256 `1648958D1563941EEE427B6222FCB4ACB283E6E2A84C76B83E1616B8C2181561`; Validation `0.934533/0.966159/0.918537`; pseudo Test `0.926526/0.961862/0.891992`; native30m `0.784859/0.879463/0.778732`; upsampled10m `0.776534/0.874212/0.770151` (IoU/F1/Kappa).
- E4: pseudo Test `0.927715/0.962502/0.896610`; native30m `0.776925/0.874460/0.772287`; upsampled10m `0.769763/0.869905/0.764588` (IoU/F1/Kappa).
- Test boundary: E0–E5 Test outputs are frozen; do not rerun any of them. Results are reporting-only and do not establish statistical significance for a single seed.
- Session-owned pending summary change: `docs/EXPERIMENTS.md` refreshes the global frozen Validation/Test tables to include E4 and E5. No unattributed changes observed.
- Verification: E5 result-record commit `820fa17`; fresh `pytest -q` passed 117 tests; `compileall` and `git diff --check` passed before the summary refresh.
- Next bounded task: create a new, explicitly seed-specific experiment design and branch in a new task; select its seed and decide whether it is a paired E0/E4/E5 replication before any training. Do not use prior Test results to select hyperparameters.
