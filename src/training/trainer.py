"""Focused E0 trainer for deterministic full-tile window training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

from src.data.windows import augment_windows, extract_windows, window_coordinates

from .checkpoint import CheckpointState, save_checkpoint
from .evaluate import evaluate_tiles
from .schedule import WarmupCosineSchedule


@dataclass(frozen=True)
class TrainerConfig:
    max_epochs: int = 100
    window_size: int = 24
    stride: int = 24
    physical_batch_size: int = 16
    effective_batch_size: int = 16
    ignore_index: int = 255
    amp: bool = True
    amp_init_scale: float = 8192.0
    amp_backoff_factor: float = 0.5
    amp_min_scale: float = 128.0
    amp_max_backoffs_per_batch: int = 6
    amp_growth_interval: int = 1_000_000
    seed: int = 42
    warmup_epochs: int = 10
    patience: int = 12
    validation_batch_size: int = 16

    def __post_init__(self) -> None:
        positive = {
            "max_epochs": self.max_epochs,
            "window_size": self.window_size,
            "stride": self.stride,
            "physical_batch_size": self.physical_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "patience": self.patience,
            "validation_batch_size": self.validation_batch_size,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("trainer sizes, epochs, and patience must be positive")
        if self.physical_batch_size > self.effective_batch_size:
            raise ValueError("physical batch cannot exceed effective batch")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")
        if self.amp_init_scale <= 0 or self.amp_growth_interval <= 0:
            raise ValueError("AMP scale and growth interval must be positive")
        if not 0.0 < self.amp_backoff_factor < 1.0:
            raise ValueError("AMP backoff factor must lie in (0,1)")
        if not 0.0 < self.amp_min_scale <= self.amp_init_scale:
            raise ValueError("AMP minimum scale must lie in (0,init_scale]")
        if (
            isinstance(self.amp_max_backoffs_per_batch, bool)
            or not isinstance(self.amp_max_backoffs_per_batch, int)
            or self.amp_max_backoffs_per_batch < 0
        ):
            raise ValueError("AMP maximum backoffs must be a non-negative integer")


class EarlyStopping:
    def __init__(self, warmup_epochs: int, patience: int) -> None:
        if warmup_epochs < 0 or patience <= 0:
            raise ValueError("invalid early-stopping parameters")
        self.warmup_epochs = int(warmup_epochs)
        self.patience = int(patience)
        self.best_metric = float("-inf")
        self.bad_epochs = 0

    def restore(self, best_metric: float, bad_epochs: int) -> None:
        self.best_metric = float(best_metric)
        self.bad_epochs = int(bad_epochs)

    def update(self, epoch: int, metric: float) -> tuple[bool, bool]:
        if not math.isfinite(metric):
            raise FloatingPointError("validation selection metric is not finite")
        improved = metric > self.best_metric
        if improved:
            self.best_metric = float(metric)
            self.bad_epochs = 0
        elif epoch > self.warmup_epochs:
            self.bad_epochs += 1
        return improved, epoch > self.warmup_epochs and self.bad_epochs >= self.patience


class E0Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Any,
        validation_dataset: Any,
        optimizer: Optimizer,
        scheduler: WarmupCosineSchedule,
        *,
        output_dir: str | Path,
        device: torch.device,
        config: TrainerConfig,
        run_config: Mapping[str, Any],
        evaluator: Callable[..., dict[str, Any]] = evaluate_tiles,
    ) -> None:
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.device = device
        self.config = config
        self.run_config = dict(run_config)
        self.evaluator = evaluator
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.amp and device.type == "cuda",
            init_scale=config.amp_init_scale,
            growth_interval=config.amp_growth_interval,
        )
        self.state = CheckpointState()

    def _raise_nonfinite(
        self,
        kind: str,
        sample_id: str,
        coordinate: Any,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        diagnostic = {
            "error": kind,
            "sample_id": sample_id,
            "window_coordinate": coordinate,
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "details": dict(details or {}),
        }
        (self.output_dir / "nonfinite_diagnostic.json").write_text(
            json.dumps(diagnostic, indent=2), encoding="utf-8"
        )
        raise FloatingPointError(f"{kind}: sample={sample_id}, window={coordinate}")

    def _gradient_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if gradient is None or torch.isfinite(gradient).all():
                continue
            finite = gradient[torch.isfinite(gradient)]
            diagnostics.append(
                {
                    "parameter": name,
                    "nan_count": int(torch.isnan(gradient).sum().item()),
                    "inf_count": int(torch.isinf(gradient).sum().item()),
                    "finite_abs_max": (
                        float(finite.abs().max().item()) if finite.numel() else None
                    ),
                }
            )
        return diagnostics

    def _append_amp_event(self, event: Mapping[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "amp_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event), sort_keys=True) + "\n")
            stream.flush()

    def train_epoch(self, epoch: int) -> dict[str, float | int]:
        self.model.train()
        generator = torch.Generator().manual_seed(self.config.seed + int(epoch))
        tile_order = torch.randperm(len(self.train_dataset), generator=generator).tolist()
        loss_sum = 0.0
        supervised_pixels = 0
        optimizer_steps = 0
        kept_windows = 0
        for tile_index in tile_order:
            sample = self.train_dataset[tile_index]
            if sample["split"] != "train":
                raise ValueError("training dataset contains a non-train record")
            images = torch.as_tensor(sample["images"])
            label = torch.as_tensor(sample["label"], dtype=torch.long)
            coordinates = window_coordinates(
                label.shape[0], label.shape[1], self.config.window_size, self.config.stride
            )
            windows, labels, kept = extract_windows(
                images,
                label,
                coordinates,
                window_size=self.config.window_size,
                skip_all_ignore=True,
                ignore_index=self.config.ignore_index,
            )
            if not kept:
                continue
            order = torch.randperm(len(kept), generator=generator)
            windows = windows[order]
            labels = labels[order]
            kept = [kept[index] for index in order.tolist()]
            doy = torch.as_tensor(sample["doy"], dtype=torch.float32)
            valid_mask = torch.as_tensor(sample["valid_mask"], dtype=torch.bool)
            for group_start in range(0, len(kept), self.config.effective_batch_size):
                group_stop = min(group_start + self.config.effective_batch_size, len(kept))
                group_images, group_labels = augment_windows(
                    windows[group_start:group_stop],
                    labels[group_start:group_stop],
                    generator=generator,
                )
                group_coordinates = kept[group_start:group_stop]
                valid_pixels = int((group_labels != self.config.ignore_index).sum().item())
                if valid_pixels == 0:
                    continue
                backoffs = 0
                while True:
                    self.optimizer.zero_grad(set_to_none=True)
                    group_loss_sum = 0.0
                    for micro_start in range(
                        0, len(group_coordinates), self.config.physical_batch_size
                    ):
                        micro_stop = min(
                            micro_start + self.config.physical_batch_size,
                            len(group_coordinates),
                        )
                        batch_images = group_images[micro_start:micro_stop].to(
                            self.device, non_blocking=True
                        )
                        batch_labels = group_labels[micro_start:micro_stop].to(
                            self.device, non_blocking=True
                        )
                        batch_size = micro_stop - micro_start
                        batch_doy = doy.unsqueeze(0).expand(batch_size, -1).to(self.device)
                        batch_mask = (
                            valid_mask.unsqueeze(0).expand(batch_size, -1).to(self.device)
                        )
                        with torch.autocast(
                            device_type=self.device.type,
                            dtype=torch.float16,
                            enabled=self.scaler.is_enabled(),
                        ):
                            logits = self.model(batch_images, batch_doy, batch_mask)
                            micro_loss_sum = F.cross_entropy(
                                logits,
                                batch_labels,
                                ignore_index=self.config.ignore_index,
                                reduction="sum",
                            )
                            objective = micro_loss_sum / valid_pixels
                        if not torch.isfinite(logits).all() or not torch.isfinite(objective):
                            self._raise_nonfinite(
                                "non-finite training loss/logits",
                                sample["sample_id"],
                                group_coordinates[micro_start],
                            )
                        self.scaler.scale(objective).backward()
                        group_loss_sum += float(micro_loss_sum.detach().item())
                    self.scaler.unscale_(self.optimizer)
                    bad_gradients = self._gradient_diagnostics()
                    if bad_gradients:
                        old_scale = float(self.scaler.get_scale())
                        can_backoff = (
                            self.scaler.is_enabled()
                            and backoffs < self.config.amp_max_backoffs_per_batch
                            and old_scale > self.config.amp_min_scale
                        )
                        new_scale = (
                            max(
                                self.config.amp_min_scale,
                                old_scale * self.config.amp_backoff_factor,
                            )
                            if can_backoff
                            else None
                        )
                        event = {
                            "event": (
                                "amp_gradient_backoff"
                                if can_backoff
                                else "amp_gradient_failure"
                            ),
                            "epoch": int(epoch),
                            "global_step": self.state.global_step,
                            "sample_id": sample["sample_id"],
                            "group_first_coordinate": list(group_coordinates[0]),
                            "group_coordinates": [
                                list(coordinate) for coordinate in group_coordinates
                            ],
                            "attempt": backoffs + 1,
                            "old_scale": old_scale,
                            "new_scale": new_scale,
                            "backoff_factor": self.config.amp_backoff_factor,
                            "min_scale": self.config.amp_min_scale,
                            "learning_rate": self.scheduler.get_last_lr()[0],
                            "group_loss_sum": group_loss_sum,
                            "valid_pixels": valid_pixels,
                            "bad_gradients": bad_gradients,
                        }
                        self._append_amp_event(event)
                        self.optimizer.zero_grad(set_to_none=True)
                        if can_backoff:
                            self.scaler.update(float(new_scale))
                            backoffs += 1
                            continue
                        self._raise_nonfinite(
                            "non-finite training gradient",
                            sample["sample_id"],
                            group_coordinates[0],
                            details=event,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.state.global_step += 1
                    optimizer_steps += 1
                    kept_windows += len(group_coordinates)
                    loss_sum += group_loss_sum
                    supervised_pixels += valid_pixels
                    break
        if supervised_pixels == 0:
            raise ValueError("training epoch has no supervised pixels")
        return {
            "loss": loss_sum / supervised_pixels,
            "supervised_pixels": supervised_pixels,
            "windows": kept_windows,
            "optimizer_steps": optimizer_steps,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

    def fit(self, initial_state: CheckpointState | None = None) -> CheckpointState:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = initial_state or CheckpointState()
        stopper = EarlyStopping(self.config.warmup_epochs, self.config.patience)
        stopper.restore(self.state.best_metric, self.state.bad_epochs)
        for epoch in range(self.state.epoch + 1, self.config.max_epochs + 1):
            self.state.epoch = epoch
            train_metrics = self.train_epoch(epoch)
            validation_metrics = self.evaluator(
                self.model,
                self.validation_dataset,
                device=self.device,
                window_size=self.config.window_size,
                stride=self.config.stride,
                window_batch_size=self.config.validation_batch_size,
                ignore_index=self.config.ignore_index,
                amp=self.config.amp,
            )
            selected = float(validation_metrics["maize_iou"])
            improved, should_stop = stopper.update(epoch, selected)
            self.state.best_metric = stopper.best_metric
            self.state.bad_epochs = stopper.bad_epochs
            record = {
                "epoch": epoch,
                "global_step": self.state.global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "best_maize_iou": self.state.best_metric,
                "bad_epochs": self.state.bad_epochs,
            }
            print(json.dumps(record, sort_keys=True), flush=True)
            with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            if improved:
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.state,
                    self.run_config,
                    metrics=record,
                )
            save_checkpoint(
                self.output_dir / "last.pt",
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.state,
                self.run_config,
                metrics=record,
            )
            if should_stop:
                break
        return self.state
