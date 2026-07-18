import pytest
import torch

from src.training.metrics import ConfusionMatrix


def test_binary_metrics_match_known_confusion_matrix() -> None:
    target = torch.tensor([[0, 0, 0], [1, 1, 1], [255, 1, 0]])
    prediction = torch.tensor([[0, 0, 1], [1, 1, 0], [1, 1, 0]])
    matrix = ConfusionMatrix(num_classes=2, ignore_index=255)

    matrix.update(prediction, target)
    metrics = matrix.compute()

    assert metrics["confusion_matrix"] == [[3, 1], [1, 3]]
    assert metrics["maize_precision"] == pytest.approx(0.75)
    assert metrics["maize_recall"] == pytest.approx(0.75)
    assert metrics["maize_f1"] == pytest.approx(0.75)
    assert metrics["maize_iou"] == pytest.approx(0.6)
    assert metrics["miou"] == pytest.approx(0.6)
    assert metrics["macro_f1"] == pytest.approx(0.75)
    assert metrics["kappa"] == pytest.approx(0.5)
    assert metrics["area_ratio"] == pytest.approx(1.0)


def test_confusion_matrices_merge_exact_counts() -> None:
    first = ConfusionMatrix(2, 255)
    second = ConfusionMatrix(2, 255)
    first.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    second.update(torch.tensor([1, 0]), torch.tensor([0, 1]))

    first.merge(second)

    assert first.compute()["confusion_matrix"] == [[1, 1], [1, 1]]
