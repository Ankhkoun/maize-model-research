"""Standalone AgriFM and selected FPN probability inference."""

import argparse
from pathlib import Path
import os


def main():
    parser = argparse.ArgumentParser(
        description="Predict maize probability with AgriFM + student FPN"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/pure_moe_lambda0.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cube", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["D_TEMPORAL_MOE_CONFIG"] = str(args.config.resolve())
    os.environ["D_TEMPORAL_MOE_STAGE"] = "student"
    import numpy as np
    import torch
    from d_temporal_moe.runtime import require_model
    from d_temporal_moe.io_utils import setup
    from d_temporal_moe.data import MEAN, STD
    from d_temporal_moe.models.network import ClassifierModel
    from d_temporal_moe.checkpoints import restore_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inference entrypoint")
    if args.output.exists():
        raise FileExistsError(args.output)
    require_model()
    setup(42)
    model = ClassifierModel().cuda()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    restore_model(model, checkpoint)
    model.eval().requires_grad_(False)
    raw = np.load(args.cube, allow_pickle=False)
    assert raw.shape == (26, 10, 256, 256) and raw.dtype == np.float32 and np.isfinite(raw).all()
    normalized = (raw * 10000 - MEAN[None, :, None, None]) / STD[None, :, None, None]
    cube = torch.from_numpy(normalized)[None].cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(dict(agrifm_cube=cube))["logits"]
    probability = logits.float().softmax(1)[0, 1].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.save(stream, probability, allow_pickle=False)


if __name__ == "__main__":
    main()
