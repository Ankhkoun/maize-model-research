from pathlib import Path

import pytest
import torch

from src.training.checkpoint import CheckpointState, load_checkpoint, save_checkpoint
from src.training.schedule import WarmupCosineSchedule


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_map_location_keeps_cpu_rng_state_compatible(tmp_path: Path) -> None:
    device = torch.device("cuda")
    model = torch.nn.Linear(1, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineSchedule(optimizer, 2, 0, 1e-3, 1e-3, 1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    path = tmp_path / "cuda.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        scaler,
        CheckpointState(epoch=1),
        {"id": "cuda-regression"},
    )

    restored = load_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        scaler,
        map_location=device,
    )

    assert restored.epoch == 1
    assert torch.get_rng_state().device.type == "cpu"
