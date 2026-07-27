# E3-ImageWPE completion handoff

- Branch: `codex/e3-image-wavelet-no-doy`; implementation commit: `ecc5aa0`.
- Design: TSViT with `temporal_position_encoding.kind: none` and content-only five-point Mexican-hat K5 WPE. The model has neither calendar DOY lookup nor learned temporal slots.
- Formal rerun: `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e3_tsvit_image_five_point_mexican_hat_k5_seed42_rerun1`; normal early stopping at epoch 27, best epoch 15/global step 49905.
- Frozen checkpoint: `best.pt` SHA256 `83F9128719C60C87CA07A044B22DCA3FF4264E23316CD0CA8D088BF7FDC9FB85`.
- Independent Validation replay: IoU/F1/Kappa `0.938460/0.968253/0.925708`, 276 tiles, result SHA256 `C5F3552B3C121BAD94BFAB686B8DB4BEBC3D2DBB7B491C24B7399315A1AFBC14`.
- One-time Test results: pseudo-label IoU/F1/Kappa `0.925872/0.961509/0.895082`; native30m `0.776375/0.874112/0.774001`; upsampled10m `0.767169/0.868247/0.764869`.
- Selection: E3 is +0.000113 IoU over E0 on frozen Validation, but below E0 on pseudo-label and both independent-reference Test IoU measures. Under the preregistered Validation-IoU selection rule, E3 is the numerical leader by 0.000113; the single-seed difference is not a significance claim, and Test results did not alter this selection.
- Test boundary: the E3 Test asset is frozen and has been run exactly once per Test entry point. Do not rerun Test without a new explicit user decision.
