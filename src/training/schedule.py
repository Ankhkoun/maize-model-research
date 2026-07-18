"""Step-based linear warmup followed by cosine learning-rate decay."""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer


class WarmupCosineSchedule:
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        start_lr: float,
        base_lr: float,
        min_lr: float,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= warmup_steps <= total_steps:
            raise ValueError("warmup_steps must lie in [0,total_steps]")
        if not 0 <= start_lr <= base_lr or not 0 <= min_lr <= base_lr:
            raise ValueError("learning rates must be non-negative and not exceed base_lr")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.start_lr = float(start_lr)
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.step_num = 0
        self._set_lr(self._lr_at(0))

    def _lr_at(self, step: int) -> float:
        step = min(max(int(step), 0), self.total_steps)
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            fraction = step / self.warmup_steps
            return self.start_lr + fraction * (self.base_lr - self.start_lr)
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps == 0:
            return self.base_lr
        progress = (step - self.warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + cosine * (self.base_lr - self.min_lr)

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self) -> float:
        self.step_num = min(self.step_num + 1, self.total_steps)
        value = self._lr_at(self.step_num)
        self._set_lr(value)
        return value

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "start_lr": self.start_lr,
            "base_lr": self.base_lr,
            "min_lr": self.min_lr,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        frozen = ("total_steps", "warmup_steps", "start_lr", "base_lr", "min_lr")
        for key in frozen:
            if state[key] != getattr(self, key):
                raise ValueError(f"schedule {key} does not match checkpoint")
        step_num = int(state["step_num"])
        if not 0 <= step_num <= self.total_steps:
            raise ValueError("checkpoint schedule step is out of range")
        self.step_num = step_num
        self._set_lr(self._lr_at(step_num))
