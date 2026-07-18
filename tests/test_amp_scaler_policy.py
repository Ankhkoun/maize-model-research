from pathlib import Path

import pytest
import torch

from src.training.schedule import WarmupCosineSchedule
from src.training.trainer import E0Trainer, TrainerConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_formal_amp_scaler_does_not_grow_during_the_run(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineSchedule(optimizer, 2, 0, 1e-3, 1e-3, 1e-3)
    config = TrainerConfig(
        max_epochs=1,
        physical_batch_size=1,
        effective_batch_size=1,
        amp=True,
        amp_init_scale=8192.0,
        amp_growth_interval=1_000_000,
    )

    trainer = E0Trainer(
        model,
        None,
        None,
        optimizer,
        scheduler,
        output_dir=tmp_path,
        device=torch.device("cuda"),
        config=config,
        run_config={"id": "amp-policy"},
    )

    assert trainer.scaler.get_scale() == 8192.0
    assert trainer.scaler.get_growth_interval() == 1_000_000
    assert trainer.config.amp_backoff_factor == 0.5
    assert trainer.config.amp_min_scale == 128.0
    assert trainer.config.amp_max_backoffs_per_batch == 6
