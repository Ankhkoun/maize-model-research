from pathlib import Path

import torch

from src.training.checkpoint import CheckpointState
from src.training.schedule import WarmupCosineSchedule
from src.training.trainer import E0Trainer, EarlyStopping, TrainerConfig


class _TinyTiles:
    def __init__(self) -> None:
        self.samples = []
        for index in range(2):
            label = torch.zeros(8, 8, dtype=torch.long)
            label[:, 4:] = 1
            self.samples.append(
                {
                    "images": label.float()[None, None].repeat(2, 1, 1, 1),
                    "label": label,
                    "doy": torch.tensor([10.0, 20.0]),
                    "valid_mask": torch.tensor([True, True]),
                    "sample_id": f"tile-{index}",
                    "region_id": "tiny",
                    "split": "train",
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Conv2d(1, 2, 1)

    def forward(self, images, doy, valid_mask):
        del doy, valid_mask
        return self.classifier(images.mean(dim=1))


def test_early_stopping_counts_only_after_warmup() -> None:
    stopper = EarlyStopping(warmup_epochs=2, patience=3)
    assert stopper.update(epoch=1, metric=0.5)[1] is False
    assert stopper.update(epoch=2, metric=0.4)[1] is False
    assert stopper.bad_epochs == 0
    assert stopper.update(epoch=3, metric=0.4)[1] is False
    assert stopper.update(epoch=4, metric=0.4)[1] is False
    assert stopper.update(epoch=5, metric=0.4)[1] is True


def test_training_epoch_uses_effective_batches_and_updates_model(tmp_path: Path) -> None:
    torch.manual_seed(42)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    schedule = WarmupCosineSchedule(optimizer, 20, 2, 1e-4, 1e-2, 1e-5)
    config = TrainerConfig(
        max_epochs=2,
        window_size=4,
        stride=4,
        physical_batch_size=2,
        effective_batch_size=4,
        amp=False,
        seed=42,
        warmup_epochs=1,
        patience=2,
    )
    trainer = E0Trainer(
        model,
        _TinyTiles(),
        None,
        optimizer,
        schedule,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config=config,
        run_config={"id": "unit"},
    )
    before = model.classifier.weight.detach().clone()

    metrics = trainer.train_epoch(epoch=1)

    assert metrics["optimizer_steps"] == 2
    assert metrics["supervised_pixels"] == 128
    assert metrics["loss"] > 0
    assert trainer.state.global_step == 2
    assert not torch.equal(before, model.classifier.weight)


def test_fit_writes_last_and_strict_best_checkpoint(tmp_path: Path) -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    schedule = WarmupCosineSchedule(optimizer, 20, 1, 1e-4, 1e-2, 1e-5)
    values = iter([0.4, 0.4])

    def evaluator(*args, **kwargs):
        del args, kwargs
        return {"maize_iou": next(values), "loss": 1.0}

    trainer = E0Trainer(
        model,
        _TinyTiles(),
        object(),
        optimizer,
        schedule,
        output_dir=tmp_path,
        device=torch.device("cpu"),
        config=TrainerConfig(
            max_epochs=2,
            window_size=4,
            stride=4,
            physical_batch_size=4,
            effective_batch_size=4,
            amp=False,
            seed=42,
            warmup_epochs=0,
            patience=5,
        ),
        run_config={"id": "unit"},
        evaluator=evaluator,
    )

    final_state = trainer.fit(CheckpointState())

    assert final_state.epoch == 2
    assert final_state.best_metric == 0.4
    assert final_state.bad_epochs == 1
    assert (tmp_path / "last.pt").is_file()
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").count("\n") == 2
