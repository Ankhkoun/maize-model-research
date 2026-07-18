import json
from pathlib import Path

import pytest
import torch

from src.training.schedule import WarmupCosineSchedule
from src.training.trainer import E0Trainer, TrainerConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _OneGroupTiles:
    def __init__(self) -> None:
        label = torch.zeros(8, 8, dtype=torch.long)
        label[:, 4:] = 1
        self.sample = {
            "images": label.float()[None, None].repeat(2, 1, 1, 1),
            "label": label,
            "doy": torch.tensor([100.0, 107.0]),
            "valid_mask": torch.tensor([True, True]),
            "sample_id": "retry-tile",
            "region_id": "retry-region",
            "split": "train",
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self.sample


class _RetryModel(torch.nn.Module):
    def __init__(self, *, fail_once: bool) -> None:
        super().__init__()
        self.classifier = torch.nn.Conv2d(1, 2, 1)
        self.fail_once = fail_once
        self.gradient_calls = 0
        self.forward_calls = 0
        self.forward_batches: list[torch.Tensor] = []
        self.classifier.weight.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient: torch.Tensor) -> torch.Tensor:
        self.gradient_calls += 1
        if not self.fail_once or self.gradient_calls == 1:
            return torch.full_like(gradient, float("inf"))
        return gradient

    def forward(self, images, doy, valid_mask):
        del doy, valid_mask
        self.forward_calls += 1
        self.forward_batches.append(images.detach().float().cpu().clone())
        return self.classifier(images.mean(dim=1))


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params) -> None:
        super().__init__(params, lr=1.0e-2)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def _build_trainer(
    tmp_path: Path,
    *,
    fail_once: bool,
    init_scale: float,
    min_scale: float,
    max_backoffs: int,
) -> tuple[E0Trainer, _RetryModel, _CountingSGD, WarmupCosineSchedule]:
    torch.manual_seed(42)
    device = torch.device("cuda")
    model = _RetryModel(fail_once=fail_once)
    optimizer = _CountingSGD(model.parameters())
    scheduler = WarmupCosineSchedule(optimizer, 10, 0, 1.0e-2, 1.0e-2, 1.0e-3)
    config = TrainerConfig(
        max_epochs=1,
        window_size=4,
        stride=4,
        physical_batch_size=4,
        effective_batch_size=4,
        amp=True,
        amp_init_scale=init_scale,
        amp_backoff_factor=0.5,
        amp_min_scale=min_scale,
        amp_max_backoffs_per_batch=max_backoffs,
        amp_growth_interval=1_000_000,
        seed=42,
        warmup_epochs=0,
        patience=1,
    )
    trainer = E0Trainer(
        model,
        _OneGroupTiles(),
        None,
        optimizer,
        scheduler,
        output_dir=tmp_path,
        device=device,
        config=config,
        run_config={"id": "amp-same-batch-retry"},
    )
    return trainer, model, optimizer, scheduler


def _read_events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_transient_gradient_overflow_replays_same_group_and_advances_once(
    tmp_path: Path,
) -> None:
    trainer, model, optimizer, scheduler = _build_trainer(
        tmp_path,
        fail_once=True,
        init_scale=8192.0,
        min_scale=128.0,
        max_backoffs=6,
    )

    metrics = trainer.train_epoch(epoch=1)

    assert metrics["optimizer_steps"] == 1
    assert metrics["windows"] == 4
    assert metrics["supervised_pixels"] == 64
    assert trainer.state.global_step == 1
    assert optimizer.step_calls == 1
    assert scheduler.step_num == 1
    assert model.forward_calls == 2
    torch.testing.assert_close(model.forward_batches[0], model.forward_batches[1])
    assert trainer.scaler.get_scale() == 4096.0
    events = _read_events(tmp_path / "amp_events.jsonl")
    assert [event["event"] for event in events] == ["amp_gradient_backoff"]
    assert events[0]["old_scale"] == 8192.0
    assert events[0]["new_scale"] == 4096.0
    assert events[0]["attempt"] == 1
    assert events[0]["sample_id"] == "retry-tile"
    assert len(events[0]["group_coordinates"]) == 4
    assert events[0]["bad_gradients"]


def test_persistent_gradient_overflow_fails_at_floor_without_state_advance(
    tmp_path: Path,
) -> None:
    trainer, model, optimizer, scheduler = _build_trainer(
        tmp_path,
        fail_once=False,
        init_scale=256.0,
        min_scale=128.0,
        max_backoffs=1,
    )
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    with pytest.raises(FloatingPointError, match="non-finite training gradient"):
        trainer.train_epoch(epoch=1)

    after = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    for name in before:
        torch.testing.assert_close(after[name], before[name], rtol=0.0, atol=0.0)
    assert optimizer.step_calls == 0
    assert scheduler.step_num == 0
    assert trainer.state.global_step == 0
    assert model.forward_calls == 2
    torch.testing.assert_close(model.forward_batches[0], model.forward_batches[1])
    events = _read_events(tmp_path / "amp_events.jsonl")
    assert [event["event"] for event in events] == [
        "amp_gradient_backoff",
        "amp_gradient_failure",
    ]
    assert events[0]["old_scale"] == 256.0
    assert events[0]["new_scale"] == 128.0
    assert events[1]["old_scale"] == 128.0
    assert events[1]["new_scale"] is None
    assert (tmp_path / "nonfinite_diagnostic.json").is_file()
