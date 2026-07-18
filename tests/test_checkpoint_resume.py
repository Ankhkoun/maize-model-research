import random

import numpy as np
import torch

from src.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from src.training.schedule import WarmupCosineSchedule


def test_resume_restores_training_objects_and_rng(tmp_path) -> None:
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = WarmupCosineSchedule(optimizer, 10, 2, 1e-8, 1e-3, 5e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = CheckpointState(epoch=3, global_step=17, best_metric=0.61, bad_epochs=2)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, schedule, scaler, state, {"id": "E0"})

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    with torch.no_grad():
        model.weight.zero_()
    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)

    restored = load_checkpoint(path, model, optimizer, schedule, scaler)

    assert restored == state
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    torch.testing.assert_close(torch.rand(3), expected_torch)
    assert schedule.step_num == 0
    assert not torch.equal(model.weight, torch.zeros_like(model.weight))


def test_checkpoint_rejects_mismatched_run_configuration(tmp_path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    schedule = WarmupCosineSchedule(optimizer, 2, 0, 1e-8, 1e-3, 5e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        schedule,
        scaler,
        CheckpointState(),
        {"seed": 42},
    )

    try:
        load_checkpoint(
            path,
            model,
            optimizer,
            schedule,
            scaler,
            expected_run_config={"seed": 7},
        )
    except ValueError as error:
        assert "configuration" in str(error)
    else:
        raise AssertionError("mismatched configuration was accepted")
