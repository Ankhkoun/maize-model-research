"""Atomic, configuration-checked training checkpoints with RNG restoration."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from .schedule import WarmupCosineSchedule


@dataclass(eq=True)
class CheckpointState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")
    bad_epochs: int = 0


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineSchedule,
    scaler: torch.amp.GradScaler,
    state: CheckpointState,
    run_config: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "training_state": asdict(state),
        "run_config": dict(run_config),
        "metrics": dict(metrics or {}),
        "rng": _rng_state(),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineSchedule,
    scaler: torch.amp.GradScaler,
    *,
    expected_run_config: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if expected_run_config is not None and payload["run_config"] != dict(expected_run_config):
        raise ValueError("checkpoint run configuration does not match")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return CheckpointState(**payload["training_state"])
