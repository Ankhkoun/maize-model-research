import numpy as np


def metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    total = max(tp + fp + fn + tn, 1)
    corn_iou = tp / max(tp + fp + fn, 1)
    noncorn_iou = tn / max(tn + fp + fn, 1)
    corn_f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    noncorn_f1 = 2 * tn / max(2 * tn + fp + fn, 1)
    oa = (tp + tn) / total
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "corn_iou": corn_iou,
        "noncorn_iou": noncorn_iou,
        "miou": (corn_iou + noncorn_iou) / 2.0,
        "corn_f1": corn_f1,
        "noncorn_f1": noncorn_f1,
        "macro_f1": (corn_f1 + noncorn_f1) / 2.0,
        "oa": oa,
        "kappa": (oa - pe) / max(1.0 - pe, 1e-12),
        "valid_pixels": int(total),
    }


def score(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    valid = target != 255
    tp = int(((prediction == 1) & (target == 1) & valid).sum())
    fp = int(((prediction == 1) & (target == 0) & valid).sum())
    fn = int(((prediction == 0) & (target == 1) & valid).sum())
    tn = int(((prediction == 0) & (target == 0) & valid).sum())
    return metrics(tp, fp, fn, tn)
