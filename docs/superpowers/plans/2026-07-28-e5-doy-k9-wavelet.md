# E5-DOY-K9WPE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and formally train the controlled E5 TSViT experiment that combines calendar DOY lookup with content-only K9 Mexican-hat WPE.

**Architecture:** Reuse the existing `FivePointMexicanHatWaveletEncoding(kernel_radius=4)` from E4. Add one frozen model YAML and one `load_formal_config` branch that accepts only E5's exact `doy_lookup` plus K9 wavelet dictionary. The existing TSViT routing then applies WPE to patch content and calendar DOY to temporal tokens without a learned temporal slot.

**Tech Stack:** Python 3.11, PyTorch 2.10 CUDA, PyYAML, pytest, PowerShell.

## Global Constraints

- Experiment ID is exactly `E5-DOY-K9WPE`; formal output directory is `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e5_tsvit_doy_nine_point_mexican_hat_k9_seed42`.
- Preserve E0/E1/E2/E3/E4 configs, frozen assets, checkpoints, and Test outputs unchanged.
- Fix Train495/Validation276/Test305, 24x24 windows, patch2, seed42, batch16, manifest, Train-only normalization, optimizer, schedule, AMP and early stopping.
- E5 must use `temporal_position_encoding: {kind: doy_lookup}` and K9 `kernel_radius: 4` with the exact frozen wavelet parameters in the approved design.
- Training and Validation replay must not load Test records. Test requires a separate explicit user authorization after replay.
- Validation maize IoU alone selects the checkpoint; do not claim statistical significance from this one-seed experiment.

---

### Task 1: Freeze the E5 formal configuration and guard

**Files:**
- Create: `configs/models/tsvit_e5_doy_wavelet_mexican_hat_k9.yaml`
- Modify: `scripts/train_e0.py:37-145`
- Create: `tests/test_e5_doy_k9_formal_config.py`

**Interfaces:**
- Consumes: `load_formal_config(path: Path) -> dict[str, Any]` and `formal_output_for(config: Mapping[str, Any]) -> Path`.
- Produces: accepted E5 config ID `E5-DOY-K9WPE`, formal output path, and exact guard failures for non-DOY or non-K9 variants.

- [ ] **Step 1: Write the failing frozen-config tests**

```python
def test_e5_formal_config_freezes_doy_plus_k9() -> None:
    config = load_formal_config(E5_PATH)
    assert config["experiment"]["id"] == "E5-DOY-K9WPE"
    assert config["model"]["temporal_position_encoding"] == {"kind": "doy_lookup"}
    assert config["model"]["wavelet"]["kernel_radius"] == 4
    assert train_e0.formal_output_for(config).name == (
        "e5_tsvit_doy_nine_point_mexican_hat_k9_seed42"
    )


def test_e5_formal_config_rejects_k5_or_missing_doy(tmp_path: Path) -> None:
    config = yaml.safe_load(E5_PATH.read_text(encoding="utf-8"))
    config["model"]["wavelet"]["kernel_radius"] = 2
    bad_k5 = tmp_path / "bad-k5.yaml"
    bad_k5.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen design"):
        load_formal_config(bad_k5)
    config["model"]["wavelet"]["kernel_radius"] = 4
    config["model"]["temporal_position_encoding"] = {"kind": "none"}
    bad_none = tmp_path / "bad-none.yaml"
    bad_none.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="calendar DOY"):
        load_formal_config(bad_none)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests\test_e5_doy_k9_formal_config.py --basetemp=.smoke\pytest-e5-config-red`

Expected: failure because the E5 YAML and formal experiment guard do not exist.

- [ ] **Step 3: Add the YAML and minimal formal guard**

```python
FORMAL_OUTPUTS["E5-DOY-K9WPE"] = Path(
    r"E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e5_tsvit_doy_nine_point_mexican_hat_k9_seed42"
)

expected_e5_wavelet = {**expected_e2_wpeonly_wavelet, "kernel_radius": 4}
if experiment_id == "E5-DOY-K9WPE":
    if config["model"].get("temporal_position_encoding", {}) != {"kind": "doy_lookup"}:
        raise ValueError("formal E5-DOY-K9WPE requires the calendar DOY lookup")
    if wavelet != expected_e5_wavelet:
        raise ValueError("formal E5-DOY-K9WPE WPE parameters differ from the frozen design")
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests\test_e5_doy_k9_formal_config.py --basetemp=.smoke\pytest-e5-config-green`

Expected: both tests pass.

- [ ] **Step 5: Commit the frozen E5 configuration**

```bash
git add configs/models/tsvit_e5_doy_wavelet_mexican_hat_k9.yaml scripts/train_e0.py tests/test_e5_doy_k9_formal_config.py
git commit -m "feat: add controlled DOY K9 wavelet experiment"
```

### Task 2: Verify E5 uses both calendar DOY and K9 content WPE

**Files:**
- Create: `tests/test_e5_doy_k9_routing.py`
- Reuse: `src/models/tsvit_segmentation.py`

**Interfaces:**
- Consumes: `TSViTSegmentation(config: Mapping[str, Any])` with a `doy_lookup` temporal encoder and `wavelet.kernel_radius=4`.
- Produces: a regression test showing calendar DOY changes logits for fixed images and valid frames, while the constructed WPE has K9 offsets.

- [ ] **Step 1: Write the failing model-routing test**

```python
def test_e5_uses_calendar_doy_and_k9_content_wavelet() -> None:
    torch.manual_seed(31)
    model = TSViTSegmentation(_small_e5_model_config()).eval()
    images = torch.randn(1, 9, 10, 8, 8)
    valid_mask = torch.ones(1, 9, dtype=torch.bool)
    with torch.no_grad():
        spring = model(images, torch.arange(90.0, 99.0).unsqueeze(0), valid_mask)
        autumn = model(images, torch.arange(250.0, 259.0).unsqueeze(0), valid_mask)
    assert model.wavelet.delta_t.tolist() == list(range(-4, 5))
    assert not torch.equal(spring, autumn)
```

- [ ] **Step 2: Run the routing test to verify it fails**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests\test_e5_doy_k9_routing.py --basetemp=.smoke\pytest-e5-routing-red`

Expected: import/configuration failure before the E5 config helper exists.

- [ ] **Step 3: Implement only the test-local E5 configuration helper**

```python
def _small_e5_model_config() -> dict:
    return {
        "image_size": 8, "patch_size": 2, "num_frames": 9,
        "num_channels": 10, "num_classes": 2, "dim": 16,
        "temporal_depth": 1, "spatial_depth": 1, "heads": 4,
        "mlp_ratio": 2.0, "dropout": 0.0, "emb_dropout": 0.0,
        "temporal_position_encoding": {"kind": "doy_lookup"},
        "wavelet": {"enabled": True, "kind": "five_point_mexican_hat",
                    "scales_init": [0.75, 1.0, 1.25],
                    "scale_bounds": [0.5, 1.5], "shifts_init": [0.0, 0.0, 0.0],
                    "shift_bounds": [-1.0, 1.0], "gamma_init": 0.001,
                    "eps": 1.0e-6, "kernel_radius": 4},
    }
```

- [ ] **Step 4: Run the routing test to verify it passes**

Run: `D:\Anaconda3\envs\cawa\python.exe -m pytest -q tests\test_e5_doy_k9_routing.py --basetemp=.smoke\pytest-e5-routing-green`

Expected: test passes, proving E5 is neither image-only nor K5.

- [ ] **Step 5: Commit the E5 routing regression test**

```bash
git add tests/test_e5_doy_k9_routing.py
git commit -m "test: verify E5 DOY and K9 routing"
```

### Task 3: Verify the clean training snapshot and launch formal E5

**Files:**
- Modify: no tracked files before launch
- Create outside repository: `E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e5_tsvit_doy_nine_point_mexican_hat_k9_seed42\*`

**Interfaces:**
- Consumes: frozen E5 YAML, `scripts/train_e0.py`, `configs/paths.local.yaml`, `manifests/xinjiang_2021_e0_e1.csv`, and `manifests/xinjiang_2021_train_normalization.json`.
- Produces: a CUDA formal training run with `run_snapshot.json`, `metrics.jsonl`, `last.pt`, and Validation-selected `best.pt`; it must not create `test_evaluation*` directories.

- [ ] **Step 1: Run full code verification**

Run:

```powershell
& 'D:\Anaconda3\envs\cawa\python.exe' -m pytest -q --basetemp=.smoke\pytest-e5-full
& 'D:\Anaconda3\envs\cawa\python.exe' -m compileall -q src scripts tests
git diff --check
```

Expected: all tests pass, compilation exits 0, and no whitespace errors.

- [ ] **Step 2: Verify CUDA availability and a new output directory**

Run:

```powershell
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
Test-Path 'E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e5_tsvit_doy_nine_point_mexican_hat_k9_seed42'
```

Expected: CUDA is available and `Test-Path` is `False`; stop instead of overwriting if the directory exists.

- [ ] **Step 3: Launch one fresh CUDA training process**

Run:

```powershell
& 'D:\Anaconda3\envs\cawa\python.exe' -u scripts\train_e0.py --paths configs\paths.local.yaml --manifest manifests\xinjiang_2021_e0_e1.csv --normalization manifests\xinjiang_2021_train_normalization.json --config configs\models\tsvit_e5_doy_wavelet_mexican_hat_k9.yaml --output-dir E:\maize_paper_workspace\06_models\retrain_outputs\maize_model_research\e5_tsvit_doy_nine_point_mexican_hat_k9_seed42 --physical-batch-size 16
```

Expected: formal script reports CUDA usage and writes Train/Validation artifacts only.

- [ ] **Step 4: Record the launched process ID and initial log state**

Run:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'tsvit_e5_doy_wavelet_mexican_hat_k9' } | Select-Object ProcessId,CreationDate,CommandLine
```

Expected: exactly one matching Python process.
