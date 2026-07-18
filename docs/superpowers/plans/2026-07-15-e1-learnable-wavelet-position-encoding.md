# E1 Learnable Wavelet Position Encoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build an Exact-style TSViT semantic segmentation baseline and an E1 variant that injects three bounded, learnable Mexican-hat wavelet bases after DOY encoding and before temporal class tokens.

**Architecture:** A standalone `LearnableWaveletPositionEncoding` computes mask-normalized pairwise responses from real DOY differences, concatenates three per-channel responses, and projects them back to the token dimension. One shared `TSViTSegmentation` implementation provides E0 and E1; E1 differs only by enabling the WPE module, so `alpha=0` restores E0 exactly.

**Tech Stack:** Python 3.13, PyTorch 2.10, pytest 8.3, PyYAML 6.0.

## Global Constraints

- Do not modify `D:\cj_swcc\_external\Exact` or `D:\cj_swcc\_external\TimeMIL`.
- Do not add CAM, TAAP, prototype learning, weak-supervision losses, quality weighting, or formal training.
- Use three Mexican-hat bases with scale initialization `[7.0, 17.5, 35.0]` days.
- Bound scales to `[3.5, 35.0]` days, shifts to `[-7.0, 7.0]` days, and support to `±42.0` days.
- WPE consumes only real temporal tokens after DOY embedding and before temporal class tokens.
- E0 and E1 must share all non-WPE architecture and parameters.

---

### Task 1: Learnable three-basis Mexican-hat module

**Files:**
- Create: `tests/test_wavelet_position_encoding.py`
- Create: `src/models/wavelet_position_encoding.py`

**Interfaces:**
- Consumes: `tokens[B,N,T,D]`, `doy[B,T]`, and boolean `valid_mask[B,T]`.
- Produces: `LearnableWaveletPositionEncoding.forward(tokens, doy, valid_mask) -> Tensor[B,N,T,D]`, plus read-only `scales` and `shifts` tensor properties for diagnostics.

- [x] **Step 1: Write failing shape, bounds, padding, invalid-input, validation, and gradient tests**

Create tests that instantiate the module with `dim=8`, assert output shape, inspect bounded scales/shifts, change only invalid token values and compare valid outputs, test all-invalid identity behavior, require finite gradients, and require `ValueError` for malformed ranks or non-boolean masks.

- [x] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_wavelet_position_encoding.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.models.wavelet_position_encoding'`.

- [x] **Step 3: Implement the minimum WPE module**

Implement:

```python
class LearnableWaveletPositionEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        scale_init_days: Sequence[float] = (7.0, 17.5, 35.0),
        scale_min_days: float = 3.5,
        scale_max_days: float = 35.0,
        shift_init_days: Sequence[float] = (0.0, 0.0, 0.0),
        shift_max_abs_days: float = 7.0,
        support_radius_days: float = 42.0,
        alpha_init: float = 0.01,
        eps: float = 1e-6,
    ) -> None: ...

    @property
    def scales(self) -> torch.Tensor: ...

    @property
    def shifts(self) -> torch.Tensor: ...

    def forward(
        self,
        tokens: torch.Tensor,
        doy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor: ...
```

Initialize raw scales directly and clamp them to the configured bounds during the forward pass; use inverse-tanh initialization for bounded shifts. Build weights as `[B,R,T,T]`, apply support and key masks, normalize by summed absolute weight, zero invalid queries, aggregate with `einsum`, concatenate scale responses, project `R*D -> D`, and return `tokens + alpha * projected_response`.

- [x] **Step 4: Run Task 1 tests to verify GREEN**

Run: `python -m pytest tests/test_wavelet_position_encoding.py -q`

Expected: all Task 1 tests pass with no warnings.

### Task 2: Shared E0/E1 TSViT segmentation model

**Files:**
- Create: `tests/test_tsvit_wpe_equivalence.py`
- Create: `src/models/tsvit_segmentation.py`
- Create: `src/models/__init__.py`

**Interfaces:**
- Consumes: a plain mapping model configuration plus `images[B,T,C,H,W]`, `doy[B,T]`, and `valid_mask[B,T]`.
- Produces: `TSViTSegmentation.forward(images, doy, valid_mask) -> logits[B,K,H,W]` and an optional `wavelet` module when enabled.

- [x] **Step 1: Write failing integration tests**

Create tests that use a small deterministic configuration (`image_size=8`, `patch_size=2`, `num_frames=5`, `num_channels=3`, `num_classes=2`, `dim=16`, one temporal and one spatial block). Verify output shape, E0/E1 equality after loading shared weights and setting E1 alpha to zero, that the WPE hook receives exactly `T` tokens before class tokens, padding-value invariance, malformed input errors, and finite forward/loss/backward values.

- [x] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_tsvit_wpe_equivalence.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'src.models.tsvit_segmentation'`.

- [x] **Step 3: Implement shared Transformer and TSViT segmentation path**

Implement focused private building blocks in `tsvit_segmentation.py`:

```python
class _TransformerBlock(nn.Module): ...
class _TransformerEncoder(nn.Module): ...

class TSViTSegmentation(nn.Module):
    def __init__(self, model_config: Mapping[str, Any]) -> None: ...

    def forward(
        self,
        images: torch.Tensor,
        doy: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor: ...
```

Use `nn.MultiheadAttention(batch_first=True)` so temporal padding is passed as `key_padding_mask`. Project scalar DOY values through a small linear embedding after scaling to `[0,1]`. Add DOY to patch tokens, optionally call WPE, reshape to `[B*N,T,D]`, prepend `K` temporal class tokens, prepend valid mask entries for those class tokens, retain the first `K` outputs, run a spatial Transformer per class, and project each patch token to `patch_size**2` logits before reconstructing `[B,K,H,W]`.

- [x] **Step 4: Run Task 2 and Task 1 tests to verify GREEN**

Run: `python -m pytest tests/test_tsvit_wpe_equivalence.py tests/test_wavelet_position_encoding.py -q`

Expected: all tests pass with no warnings.

### Task 3: E0/E1 configuration and complete verification

**Files:**
- Create: `configs/models/tsvit_baseline.yaml`
- Create: `configs/models/tsvit_wpe_basic.yaml`

**Interfaces:**
- Consumes: YAML mappings matching `TSViTSegmentation` constructor fields.
- Produces: an E0 config with `wavelet.enabled: false` and an E1 config with identical non-WPE settings plus the approved WPE bounds.

- [x] **Step 1: Add a failing config-consistency test**

Extend `tests/test_tsvit_wpe_equivalence.py` to load both YAML files, remove only the `wavelet` mapping, and assert the remaining model mappings are equal. Instantiate both models and run a small forward pass.

- [x] **Step 2: Run the config test to verify RED**

Run: `python -m pytest tests/test_tsvit_wpe_equivalence.py -q`

Expected: failure because one or both configuration files do not exist.

- [x] **Step 3: Add matching E0 and E1 YAML configurations**

Use 2021 Xinjiang contract defaults (`image_size=256`, `patch_size=2`, `num_frames=26`, `num_channels=10`, `num_classes=2`) and identical Transformer dimensions, dropout, and seed. E1 adds the approved three-basis WPE parameters; E0 disables WPE.

- [x] **Step 4: Run full verification**

Run: `python -m pytest -q`

Expected: all tests pass, with zero failures and zero warnings.

Run: `python -m compileall -q src tests`

Expected: exit code 0.

Run: `git diff --check`

Expected: exit code 0 and no output.

- [x] **Step 5: Review the final diff without publication**

Run: `git status --short` and `git diff -- src tests configs docs/superpowers`.

Expected: only the approved design, implementation plan, E0/E1 source, tests, and model configuration files are present. Do not stage, commit, or push.
