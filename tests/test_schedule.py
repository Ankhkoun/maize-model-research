import pytest
import torch

from src.training.schedule import WarmupCosineSchedule


def test_warmup_cosine_schedule_hits_frozen_endpoints() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=99.0)
    schedule = WarmupCosineSchedule(
        optimizer,
        total_steps=10,
        warmup_steps=2,
        start_lr=1e-8,
        base_lr=1e-3,
        min_lr=5e-6,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-8)
    schedule.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx((1e-8 + 1e-3) / 2)
    schedule.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    for _ in range(8):
        schedule.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)


def test_schedule_round_trip_restores_step_and_learning_rate() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    schedule = WarmupCosineSchedule(optimizer, 20, 4, 1e-8, 1e-3, 5e-6)
    for _ in range(7):
        schedule.step()
    state = schedule.state_dict()

    other_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.tensor(2.0))], lr=0.1)
    other = WarmupCosineSchedule(other_optimizer, 20, 4, 1e-8, 1e-3, 5e-6)
    other.load_state_dict(state)

    assert other.step_num == 7
    assert other_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
