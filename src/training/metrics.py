"""Exact confusion-count metrics for binary semantic segmentation."""

from __future__ import annotations

import torch


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be at least two")
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = torch.as_tensor(prediction).detach().to(device="cpu", dtype=torch.long)
        target = torch.as_tensor(target).detach().to(device="cpu", dtype=torch.long)
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        valid = target != self.ignore_index
        prediction = prediction[valid]
        target = target[valid]
        if target.numel() == 0:
            return
        if ((target < 0) | (target >= self.num_classes)).any():
            raise ValueError("target contains an out-of-range class")
        if ((prediction < 0) | (prediction >= self.num_classes)).any():
            raise ValueError("prediction contains an out-of-range class")
        bins = torch.bincount(
            target * self.num_classes + prediction,
            minlength=self.num_classes**2,
        )
        self.matrix += bins.reshape(self.num_classes, self.num_classes)

    def merge(self, other: "ConfusionMatrix") -> None:
        if (
            self.num_classes != other.num_classes
            or self.ignore_index != other.ignore_index
        ):
            raise ValueError("confusion matrices must share class and ignore settings")
        self.matrix += other.matrix

    def compute(self) -> dict[str, float | list[list[int]]]:
        matrix = self.matrix.double()
        total = float(matrix.sum().item())
        diagonal = matrix.diag()
        target_count = matrix.sum(dim=1)
        prediction_count = matrix.sum(dim=0)
        union = target_count + prediction_count - diagonal
        per_class_iou = [
            _ratio(float(diagonal[index]), float(union[index]))
            for index in range(self.num_classes)
        ]
        per_class_f1 = [
            _ratio(
                2.0 * float(diagonal[index]),
                float(target_count[index] + prediction_count[index]),
            )
            for index in range(self.num_classes)
        ]
        maize = 1
        tp = float(matrix[maize, maize])
        fp = float(prediction_count[maize] - matrix[maize, maize])
        fn = float(target_count[maize] - matrix[maize, maize])
        expected_agreement_count = float((target_count * prediction_count).sum())
        kappa_denominator = total * total - expected_agreement_count
        kappa = _ratio(
            total * float(diagonal.sum()) - expected_agreement_count,
            kappa_denominator,
        )
        return {
            "overall_accuracy": _ratio(float(diagonal.sum()), total),
            "maize_precision": _ratio(tp, tp + fp),
            "maize_recall": _ratio(tp, tp + fn),
            "maize_f1": _ratio(2.0 * tp, 2.0 * tp + fp + fn),
            "maize_iou": per_class_iou[maize],
            "miou": sum(per_class_iou) / self.num_classes,
            "macro_f1": sum(per_class_f1) / self.num_classes,
            "kappa": kappa,
            "area_ratio": _ratio(float(prediction_count[maize]), float(target_count[maize])),
            "confusion_matrix": self.matrix.tolist(),
        }
