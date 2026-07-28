# E4-ImageWPE-K9 completion handoff

- Branch: `codex/e4-image-wavelet-k9`; implementation commit: `f4bc607`.
- Design: strict image-only TSViT with `temporal_position_encoding.kind: none` and content-only Mexican-hat K9 WPE (`kernel_radius=4`, offsets -4 through +4). It has neither calendar DOY lookup nor learned temporal slots.
- Formal training: normal early stopping at epoch 23; frozen best epoch 11 / global step 36597.
- Frozen checkpoint: `best.pt` SHA256 `E30E912C7ECEA90FAEFCCCF8444849F7F36A1137084FDDC9F70EC0F1204ABE14`.
- Independent Validation replay: 276 tiles; IoU/F1/Kappa `0.938988/0.968534/0.926002`, exactly matching epoch 11.
- One-time Test results: pseudo-label IoU/F1/Kappa `0.927715/0.962502/0.896610`; native30m `0.776925/0.874460/0.772287`; upsampled10m `0.769763/0.869905/0.764588`.
- Selection: E4 is the numerical single-seed Validation-IoU leader (+0.000527 vs E3; +0.000640 vs E0). Its pseudo-label Test IoU is +0.000208 above E0, while both independent annual-reference IoU values remain below E0. Test is reporting-only and does not alter the frozen Validation-based selection rule. Do not claim statistical significance from this one-seed comparison.
