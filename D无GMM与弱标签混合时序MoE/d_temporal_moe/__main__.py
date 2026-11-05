"""Run only the explicit requested stage of the single final model."""

from pathlib import Path
import argparse
import importlib
import os

ACTIONS = {
    "check-data": ("teacher", "d_temporal_moe.training.warmup", "check_data"),
    "warmup": ("teacher", "d_temporal_moe.training.warmup", "train"),
    "neighbors": ("teacher", "d_temporal_moe.training.neighbors", "prepare"),
    "train-teacher": ("teacher", "d_temporal_moe.training.teacher", "train"),
    "prepare-experts": ("student", "d_temporal_moe.training.experts", "prepare"),
    "fit-experts": ("student", "d_temporal_moe.training.experts", "fit_experts"),
    "make-targets": ("student", "d_temporal_moe.training.experts", "generate_targets"),
    "train-student": ("student", "d_temporal_moe.training.student", "train"),
    "seal": ("student", "d_temporal_moe.training.student", "seal"),
    "test": ("student", "d_temporal_moe.evaluation.test", "test"),
}


def main():
    parser = argparse.ArgumentParser(description="D + dual temporal MoE")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/pure_moe_lambda0.json",
    )
    parser.add_argument("action", choices=["stages", "show-config", *ACTIONS])
    args = parser.parse_args()
    if args.action == "stages":
        for action, (stage, module, function) in ACTIONS.items():
            print(f"{action:18} {stage:8} {module}.{function}")
        return
    if args.action == "show-config":
        print(args.config.read_text(encoding="utf-8"))
        return
    stage, module, function = ACTIONS[args.action]
    os.environ["D_TEMPORAL_MOE_CONFIG"] = str(args.config.resolve())
    os.environ["D_TEMPORAL_MOE_STAGE"] = stage
    import torch

    if args.action not in ("check-data", "seal") and (not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA is required for full training/evaluation; use tests/ for CPU mathematical checks"
        )
    from d_temporal_moe.runtime import require_inputs

    require_inputs()
    getattr(importlib.import_module(module), function)()


if __name__ == "__main__":
    main()
