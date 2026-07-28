# E5-DOY-K9WPE completion handoff

- Branch: `codex/e5-doy-k9-wavelet`; configuration commit: `a555757`; routing-test commit: `7f322ed`.
- Design: E0 calendar DOY lookup plus content-only Mexican-hat K9 WPE (`kernel_radius=4`, offsets -4 through +4); no learned temporal slot.
- Formal training: normal early stopping at epoch 26; frozen best epoch 14 / global step 46578.
- Frozen checkpoint: `best.pt` SHA256 `1648958D1563941EEE427B6222FCB4ACB283E6E2A84C76B83E1616B8C2181561`.
- Independent Validation replay: 276 tiles; IoU/F1/Kappa `0.934533/0.966159/0.918537`, exactly matching epoch 14.
- One-time Test results: pseudo-label IoU/F1/Kappa `0.926526/0.961862/0.891992`; native30m `0.784859/0.879463/0.778732`; upsampled10m `0.776534/0.874212/0.770151`.
- Interpretation: E5 does not exceed E0 on Validation or pseudo-label Test. On the primary native30m reference, it is only -0.000238 IoU and -0.000149 F1 below E0, but -0.003290 Kappa. It improves native30m IoU over E4 by +0.007934, showing that restoring calendar DOY largely recovers the external-reference gap; this single-seed result is not a significance claim. Test is reporting-only and does not change the Validation-based selection rule.
