# Fixed-window Mexican-hat WPE

This document describes the fixed-window wavelet position encoding used by the
USA TSViT experiments E12, E13, and E16. The implementation is in
`src/models/fixed_window_wpe.py` and its integration contract is tested in
`tests/test_e12_fixed_window_wpe.py`.

## Definition

For each fixed temporal offset `delta` and scale `sigma`, the nominal kernel is

```text
u = delta / sigma
psi(u) = (1 - u^2) exp(-u^2 / 2)
```

Each nominal window is made zero-mean and L1-normalized. The radii, sigmas, and
shifts are buffers rather than learned parameters. All published configurations
use shift `0`.

At sequence boundaries and missing frames, only in-range valid taps are kept.
Those taps are made zero-mean and L1-normalized again for each query. A query
with fewer than two valid taps receives a zero residual. This is the
`masked_renormalized` boundary rule.

The three wavelet responses are fused with a per-channel softmax matrix
`softmax(raw_gates, dim=0)` of shape `[3, D]`. The result is normalized and
injected as a residual:

```text
y = x + gamma * LayerNorm(sum_s gate_s * WPE_s(LayerNorm(x)))
```

`positive_bounded` uses `gamma = gamma_max * sigmoid(raw_gamma)`.
`fixed_positive` stores gamma as a non-trainable buffer. The published initial
or fixed value is `0.02`, with maximum `0.20` for the learnable form.

## TSViT injection point

The module receives only patch content tokens with shape `[B, N, T, D]`.
Calendar or cyclic DOY tokens are added after WPE, and temporal class tokens are
concatenated after that. Therefore neither DOY embeddings nor class tokens are
convolved by the wavelet. Invalid frames are masked and the fixed-window path is
forced to FP32.

For cyclic DOY, TSViT projects `[sin(2*pi*doy/365), cos(2*pi*doy/365)]` through
a bias-free linear layer and applies amplitude `0.10`. This absolute seasonal
signal remains separate from the content-only WPE residual.

## Frozen configurations

| Experiment | Temporal position | Windows | Radii | Sigmas (time steps) | Gamma |
| --- | --- | --- | --- | --- | --- |
| E12 learnable | none | K5/K13/K25 | 2/6/12 | 1/3/6 | bounded, init 0.02 |
| E12 fixed | none | K5/K13/K25 | 2/6/12 | 1/3/6 | fixed 0.02 |
| E13 | cyclic DOY | K5/K13/K25 | 2/6/12 | 1/3/6 | fixed 0.02 |
| E16 | cyclic DOY | K5/K9/K17 | 2/4/8 | 1/2/4 | fixed 0.02 |

All four configurations use seed 42, Nebraska 400-tile Train, Colorado 321-tile
Validation, 24x24 windows, patch size 2, 26 frames, 10 channels, dimension 128,
temporal/spatial depth 4/4, 4 heads, batch 32, AdamW, ordinary augmented
pseudo-label cross entropy, full FP32, and no AMP. Checkpoints are selected only
by Colorado Validation maize IoU. The training loader selects only Train and
Validation records and records `test_records_loaded=0`.

## Recorded results

| Experiment | Colorado Validation maize IoU | Iowa all-valid maize IoU |
| --- | ---: | ---: |
| E12 learnable gamma | 0.940772 | 0.743701 |
| E12 fixed gamma | 0.934467 | 0.789664 |
| E13 | 0.936442 | 0.786846 |
| E16 | 0.936372 | 0.763950 |
| E14 cyclic DOY only (reference) | 0.937444 | 0.793314 |

Iowa was evaluated only after each checkpoint was frozen. These Test values are
reporting-only and must not be used to select windows, gamma, checkpoints, or
other hyperparameters.

## Verification

Run the fixed-window contract tests from the repository root:

```powershell
python -m pytest -q tests/test_e12_fixed_window_wpe.py
```

Formal runs also write `usa_fixed_window_wpe_metrics.jsonl`. Each epoch records
the effective residual ratio, gamma, the three window fusion summaries, learning
rates, peak GPU memory, epoch duration, early-stopping state, and checkpoint
status.
