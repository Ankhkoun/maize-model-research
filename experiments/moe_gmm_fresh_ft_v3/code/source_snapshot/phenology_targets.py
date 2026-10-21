"""Weak phenology targets derived from per-patch spectral time series.

These targets are auxiliary route supervision only.  They are never treated as
CDL labels and are masked to high-quality pseudo-corn patches.
"""

from __future__ import annotations

import numpy as np


NUM_FRAMES = 26
NUM_PHASES = 4
PATCH_SIDE = 128
PATCHES_PER_WINDOW = 1024
WINDOWS_PER_TILE = 16


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float32),
                    where=np.abs(denominator) > 1.0e-6)


def _tile_to_windows(value: np.ndarray) -> np.ndarray:
    """Convert [T,H,W,...] patch-grid data to [16,1024,T,...]."""
    if value.ndim not in (3, 4, 5) or value.shape[1:3] != (PATCH_SIDE, PATCH_SIDE):
        raise ValueError(f"expected [T,128,128,...], got {value.shape}")
    tail = value.shape[3:]
    tiled = value.reshape(NUM_FRAMES, 4, 32, 4, 32, *tail)
    tiled = tiled.transpose(1, 3, 2, 4, 0, *range(5, 5 + len(tail)))
    return tiled.reshape(WINDOWS_PER_TILE, PATCHES_PER_WINDOW, NUM_FRAMES, *tail)


def windows_to_tile_phase(value: np.ndarray) -> np.ndarray:
    """Convert [16,1024,T,K] phase data to [T,K,128,128]."""
    value = np.asarray(value)
    if value.shape != (WINDOWS_PER_TILE, PATCHES_PER_WINDOW, NUM_FRAMES, NUM_PHASES):
        raise ValueError(f"expected [16,1024,26,4], got {value.shape}")
    tile = value.reshape(4, 4, 32, 32, NUM_FRAMES, NUM_PHASES)
    return tile.transpose(4, 5, 0, 2, 1, 3).reshape(NUM_FRAMES, NUM_PHASES, PATCH_SIDE, PATCH_SIDE)


def tile_to_windows_phase(value: np.ndarray) -> np.ndarray:
    """Convert [T,K,128,128] phase data to [16,1024,T,K]."""
    value = np.asarray(value)
    if value.shape != (NUM_FRAMES, NUM_PHASES, PATCH_SIDE, PATCH_SIDE):
        raise ValueError(f"expected [26,4,128,128], got {value.shape}")
    tile = value.transpose(2, 3, 0, 1).reshape(4, 32, 4, 32, NUM_FRAMES, NUM_PHASES)
    return tile.transpose(0, 2, 1, 3, 4, 5).reshape(WINDOWS_PER_TILE, PATCHES_PER_WINDOW, NUM_FRAMES, NUM_PHASES)


def windows_to_tile_scalar(value: np.ndarray) -> np.ndarray:
    """Convert [16,1024,T] scalar data to [T,128,128]."""
    value = np.asarray(value)
    if value.shape != (WINDOWS_PER_TILE, PATCHES_PER_WINDOW, NUM_FRAMES):
        raise ValueError(f"expected [16,1024,26], got {value.shape}")
    tile = value.reshape(4, 4, 32, 32, NUM_FRAMES)
    return tile.transpose(4, 0, 2, 1, 3).reshape(NUM_FRAMES, PATCH_SIDE, PATCH_SIDE)


def tile_to_windows_scalar(value: np.ndarray) -> np.ndarray:
    """Convert [T,128,128] scalar data to [16,1024,T]."""
    value = np.asarray(value)
    if value.shape != (NUM_FRAMES, PATCH_SIDE, PATCH_SIDE):
        raise ValueError(f"expected [26,128,128], got {value.shape}")
    tile = value.transpose(1, 2, 0).reshape(4, 32, 4, 32, NUM_FRAMES)
    return tile.transpose(0, 2, 1, 3, 4).reshape(WINDOWS_PER_TILE, PATCHES_PER_WINDOW, NUM_FRAMES)


def build_phase_targets(raw_cube: np.ndarray, observed_support: np.ndarray,
                        pseudo_class: np.ndarray, pseudo_class_valid: np.ndarray,
                        *, min_observed: int = 6, min_amplitude: float = 0.08,
                        transition_temperature: float = 0.045) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Build soft four-phase targets from reflectance curves.

    ``raw_cube`` is reflectance in [0,1], ``observed_support`` is [16,1024,26],
    and pseudo patch fields are [16,1024].  The target is valid only for
    pseudo-corn patches with enough direct observations and a measurable
    seasonal signal.  Stage names remain data-driven until post-training
    diagnostics map the learned ordered experts to curve landmarks.
    """
    raw = np.asarray(raw_cube, dtype=np.float32)
    if raw.shape != (NUM_FRAMES, 10, 256, 256):
        raise ValueError(f"raw cube must be [26,10,256,256], got {raw.shape}")
    support = np.asarray(observed_support, dtype=np.float32)
    if support.shape != (WINDOWS_PER_TILE, PATCHES_PER_WINDOW, NUM_FRAMES):
        raise ValueError(f"support must be [16,1024,26], got {support.shape}")
    pclass = np.asarray(pseudo_class)
    pvalid = np.asarray(pseudo_class_valid).astype(bool)
    if pclass.shape != (WINDOWS_PER_TILE, PATCHES_PER_WINDOW) or pvalid.shape != pclass.shape:
        raise ValueError("pseudo patch fields must be [16,1024]")

    # Bands use the frozen cube order B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12.
    b4 = raw[:, 2]
    b5 = raw[:, 3]
    b8 = raw[:, 6]
    b8a = raw[:, 7]
    b11 = raw[:, 8]
    ndvi = _safe_ratio(b8 - b4, b8 + b4)
    ndre = _safe_ratio(b8a - b5, b8a + b5)
    ndwi = _safe_ratio(b8 - b11, b8 + b11)
    signal = 0.50 * ndvi + 0.35 * ndre + 0.15 * ndwi
    # Aggregate each 2x2 token before estimating its temporal landmarks.
    signal = signal.reshape(NUM_FRAMES, PATCH_SIDE, 2, PATCH_SIDE, 2).mean(axis=(2, 4))
    signal = tile_to_windows_scalar(signal)
    finite = np.isfinite(signal)
    observed = support > 0.0
    usable = finite & observed
    safe_signal = np.where(usable, signal, np.nan)
    count = usable.sum(axis=-1)
    baseline = np.nanmin(safe_signal, axis=-1)
    peak_value = np.nanmax(safe_signal, axis=-1)
    baseline = np.nan_to_num(baseline, nan=0.0)
    peak_value = np.nan_to_num(peak_value, nan=0.0)
    amplitude = np.maximum(peak_value - baseline, 0.0)
    normalized = np.clip((np.nan_to_num(signal, nan=0.0) - baseline[..., None]) /
                         np.maximum(amplitude[..., None], 1.0e-6), 0.0, 1.0)
    normalized = np.where(usable, normalized, -np.inf)
    peak_index = normalized.argmax(axis=-1)
    time = np.arange(NUM_FRAMES, dtype=np.float32)[None, None, :]
    before_peak = time <= peak_index[..., None]
    green_hits = usable & before_peak & (normalized >= 0.20)
    green_index = np.where(green_hits, time, NUM_FRAMES).min(axis=-1)
    first_valid = np.where(usable, time, NUM_FRAMES).min(axis=-1)
    green_index = np.where(green_index < NUM_FRAMES, green_index, first_valid)
    after_peak = time >= peak_index[..., None]
    senescence_hits = usable & after_peak & (normalized <= 0.80)
    senescence_index = np.where(senescence_hits, time, NUM_FRAMES).min(axis=-1)
    last_valid = np.where(usable, time, 0).max(axis=-1)
    senescence_index = np.where(senescence_index < NUM_FRAMES, senescence_index, last_valid)
    senescence_index = np.maximum(senescence_index, peak_index)

    # Piecewise normalized progress: green-up -> peak -> senescence.
    green = green_index[..., None]
    peak = peak_index[..., None].astype(np.float32)
    senescence = senescence_index[..., None]
    denom_a = np.maximum(peak - green, 1.0)
    denom_b = np.maximum(senescence - peak, 1.0)
    progress = np.where(
        time <= green,
        0.0,
        np.where(time <= peak, (time - green) / denom_a * (1.0 / 3.0),
                 np.where(time <= senescence, 1.0 / 3.0 + (time - peak) / denom_b * (1.0 / 3.0), 1.0)),
    )
    centers = np.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float32)
    distances = (progress[..., None] - centers.reshape(1, 1, 1, NUM_PHASES)) ** 2
    target = np.exp(-distances / float(transition_temperature))
    target /= np.maximum(target.sum(axis=-1, keepdims=True), 1.0e-8)

    patch_confidence = np.clip(amplitude / float(min_amplitude), 0.0, 1.0)
    patch_confidence *= np.clip(count / float(NUM_FRAMES), 0.0, 1.0)
    patch_valid = (pclass == 1) & pvalid & (count >= int(min_observed)) & (amplitude >= float(min_amplitude))
    phase_valid = usable & patch_valid[..., None]
    confidence = phase_valid.astype(np.float32) * patch_confidence[..., None].astype(np.float32)
    target = target.astype(np.float32)
    target[~phase_valid] = 0.0
    report = {
        "patch_count": float(patch_valid.size),
        "valid_patch_fraction": float(patch_valid.mean()),
        "valid_frame_fraction": float(phase_valid.mean()),
        "mean_amplitude": float(amplitude[patch_valid].mean()) if bool(patch_valid.any()) else 0.0,
        "mean_support": float(support[phase_valid].mean()) if bool(phase_valid.any()) else 0.0,
    }
    return target, phase_valid, confidence, report


__all__ = [
    "build_phase_targets", "tile_to_windows_phase", "windows_to_tile_phase",
    "tile_to_windows_scalar", "windows_to_tile_scalar",
]
