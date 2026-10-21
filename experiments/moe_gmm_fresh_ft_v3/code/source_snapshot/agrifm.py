"""AgriFM Video Swin loader and full-resolution FPN classification branch."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _registry_shims() -> None:
    # The released AgriFM source imports these two small timm utilities, but
    # the remote CUDA runtime intentionally does not install the full timm
    # package.  Keep their standard behavior local to this independent model
    # package so the pretrained backbone does not depend on global packages.
    if "timm.models.layers" not in sys.modules:
        timm = sys.modules.setdefault("timm", types.ModuleType("timm"))
        timm.__path__ = []
        models = sys.modules.setdefault("timm.models", types.ModuleType("timm.models"))
        models.__path__ = []
        layers = types.ModuleType("timm.models.layers")

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x: Tensor) -> Tensor:
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
                return x * mask

        layers.DropPath = DropPath
        layers.trunc_normal_ = nn.init.trunc_normal_
        sys.modules["timm.models.layers"] = layers

    if "einops" not in sys.modules:
        einops = types.ModuleType("einops")

        def rearrange(x: Tensor, pattern: str, **sizes: int) -> Tensor:
            pattern = " ".join(pattern.split())
            if pattern == "b c d h w -> b d h w c":
                return x.permute(0, 2, 3, 4, 1)
            if pattern == "b d h w c -> b c d h w":
                return x.permute(0, 4, 1, 2, 3)
            if pattern == "b c t h w -> (b c) 1 t h w":
                b, c, t, h, w = x.shape
                return x.reshape(b * c, 1, t, h, w)
            if pattern == "(b c) d t h w -> b c d t h w":
                b, c = int(sizes["b"]), int(sizes["c"])
                return x.reshape(b, c, *x.shape[1:])
            if pattern == "b c d t h w -> (b c) d t h w":
                b, c, d, t, h, w = x.shape
                return x.reshape(b * c, d, t, h, w)
            if pattern == "(b c) 1 t h w -> b c 1 t h w":
                b, c = int(sizes["b"]), int(sizes["c"])
                return x.reshape(b, c, 1, *x.shape[2:])
            if pattern == "n c d h w -> n d h w c":
                return x.permute(0, 2, 3, 4, 1)
            if pattern == "n d h w c -> n c d h w":
                return x.permute(0, 4, 1, 2, 3)
            raise NotImplementedError(f"AgriFM einops rearrange pattern: {pattern}")

        def reduce(x: Tensor, pattern: str, reduction: str, **sizes: int) -> Tensor:
            pattern = " ".join(pattern.split())
            if pattern == "b c (t s) h w -> b c t h w" and reduction == "mean":
                s = int(sizes["s"])
                b, c, ts, h, w = x.shape
                if ts % s:
                    raise ValueError(f"temporal feature length {ts} is not divisible by s={s}")
                return x.reshape(b, c, ts // s, s, h, w).mean(dim=3)
            raise NotImplementedError(f"AgriFM einops reduce pattern: {pattern}/{reduction}")

        einops.rearrange = rearrange
        einops.reduce = reduce
        sys.modules["einops"] = einops

    class Registry:
        def register_module(self, *args: Any, **kwargs: Any):
            def decorate(cls):
                return cls
            return decorate

    registry = Registry()
    for name in ("mmseg", "mmseg.models", "mmseg.models.builder", "mmseg.registry",
                 "mmseg.registry.registry"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
    sys.modules["mmseg.models.builder"].BACKBONES = registry
    for name in ("mmseg.registry", "mmseg.registry.registry"):
        sys.modules[name].MODELS = registry
        sys.modules[name].TRANSFORMS = registry

    if "mmengine.model" not in sys.modules:
        module = types.ModuleType("mmengine.model")

        class BaseModule(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

        class BaseModel(BaseModule):
            pass

        module.BaseModule = BaseModule
        module.BaseModel = BaseModel
        sys.modules["mmengine.model"] = module
    if "mmengine.runner" not in sys.modules:
        module = types.ModuleType("mmengine.runner")
        module.load_checkpoint = lambda *args, **kwargs: None
        sys.modules["mmengine.runner"] = module
    if "mmengine" not in sys.modules:
        module = types.ModuleType("mmengine")
        module.__path__ = []
        sys.modules["mmengine"] = module


def _source_module(source_root: str | Path):
    root = Path(source_root).expanduser().resolve()
    source = root / "AgriFM" / "models" / "video_swin_transformer.py"
    if not source.is_file():
        raise FileNotFoundError(f"AgriFM Video Swin source missing: {source}")
    name = "_af2_usa_video_swin"
    if name in sys.modules:
        return sys.modules[name]
    _registry_shims()
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import AgriFM Video Swin source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_state(path: str | Path) -> dict[str, Tensor]:
    payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(payload, dict) and key in payload and isinstance(payload[key], dict):
            payload = payload[key]
            break
    if not isinstance(payload, dict):
        raise ValueError("AgriFM checkpoint does not contain a state dictionary")
    return {str(k): v for k, v in payload.items() if isinstance(v, Tensor)}


def _canonical_key(key: str) -> str:
    while key.startswith("module.") or key.startswith("encoder."):
        key = key.split(".", 1)[1]
    key = re.sub(r"^S2_patch_emd\.", "patch_embed.proj.", key)
    key = re.sub(r"^patch_emd\.", "patch_embed.proj.", key)
    if key == "patch_embed.proj.weights":
        key = "patch_embed.proj.weight"
    return key


def load_pretrained(encoder: nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    """Load compatible AgriFM encoder tensors and reject mostly-random loads."""
    source = _checkpoint_state(checkpoint)
    target = encoder.state_dict()
    mapped: dict[str, Tensor] = {}
    unexpected: list[str] = []
    shape_mismatch: list[dict[str, Any]] = []
    adapted_keys: list[dict[str, Any]] = []
    for original, value in source.items():
        key = _canonical_key(original)
        if key == "patch_embed.proj.weight" and key in target:
            wanted = target[key].shape
            if value.ndim == 5 and value.shape[:2] == wanted[:2] and value.shape[-3:] == (4, 4, 4) and wanted[-3:] == (4, 2, 2):
                original_shape = list(value.shape)
                value = F.interpolate(
                    value.float().reshape(-1, 1, *value.shape[-2:]), size=(2, 2),
                    mode="bilinear", align_corners=False,
                ).reshape(*value.shape[:3], 2, 2).to(dtype=value.dtype)
                adapted_keys.append({"key": key, "from": original_shape, "to": list(wanted),
                                     "method": "bilinear_spatial_kernel_resize"})
        if key not in target:
            unexpected.append(original)
        elif target[key].shape != value.shape:
            shape_mismatch.append({"key": key, "checkpoint": list(value.shape),
                                   "model": list(target[key].shape)})
        else:
            mapped[key] = value
    result = encoder.load_state_dict(mapped, strict=False)
    parameter_total = sum(p.numel() for p in encoder.parameters())
    parameter_loaded = sum(encoder.state_dict()[k].numel() for k in mapped
                           if k in dict(encoder.named_parameters()))
    coverage = parameter_loaded / max(parameter_total, 1)
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "loaded_parameter_fraction": coverage,
        "loaded_tensors": len(mapped),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "adapted_keys": adapted_keys,
    }
    if coverage < 0.90:
        raise RuntimeError(f"AgriFM pretrained load covered only {coverage:.1%} of parameters")
    return report


class ConvNormAct(nn.Sequential):
    def __init__(self, channels: int, width: int):
        super().__init__(nn.Conv2d(channels, width, 1, bias=False),
                         nn.GroupNorm(8, width), nn.GELU())


class NativeFPN256(nn.Module):
    """Three-level FPN over the native 64/32/16 AgriFM feature maps."""

    def __init__(self, width: int = 128, classes: int = 2):
        super().__init__()
        self.lateral = nn.ModuleList([ConvNormAct(1024, width) for _ in range(3)])
        self.refine = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.GELU(),
        )
        self.classifier = nn.Conv2d(width, classes, 1)

    def forward(self, f1: Tensor, f2: Tensor, f3: Tensor) -> Tensor:
        x1, x2, x3 = (layer(x) for layer, x in zip(self.lateral, (f1, f2, f3)))
        x = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False) + x2
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False) + x1
        x = self.refine(x)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        return self.classifier(x)


class AgriFMClassifier(nn.Module):
    """Official AgriFM patch embed/Swin encoder with a new 256px FPN head."""

    def __init__(self, source_root: str | Path, checkpoint: str | Path,
                 *, use_checkpointing: bool = True):
        super().__init__()
        source = _source_module(source_root)
        # The released cropland-mapping configuration uses the static
        # SwinPatchEmbed3D. Keeping it static avoids random dynamic-generator
        # tensors that are absent from the AgriFM checkpoint.
        self.patch_embed = source.SwinPatchEmbed3D(
            patch_size=(4, 2, 2), in_chans=10, embed_dim=128,
        )
        self.backbone = source.SwinTransformer3D(
            pretrained=None, pretrained2d=False, patch_size=(4, 2, 2),
            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
            window_size=(8, 7, 7), out_indices=(0, 1, 2, 3), mlp_ratio=4.0,
            qkv_bias=True, drop_path_rate=0.2, patch_norm=False, frozen_stages=-1,
            use_checkpoint=use_checkpointing, downsample_steps=((2, 2, 2),) * 4,
            feature_fusion="cat", mean_frame_down=True,
        )
        self.load_report = load_pretrained(self, checkpoint)
        self.decoder = NativeFPN256()

    @staticmethod
    def _f1_groups(flat: Tensor) -> Tensor:
        if flat.ndim != 4 or flat.shape[1:] != (1024, 64, 64):
            raise ValueError(f"F1 must be [B,1024,64,64], got {tuple(flat.shape)}")
        batch = flat.shape[0]
        # AgriFM flattens [B,C,T,H,W] as [B,C*T,H,W].
        return flat.reshape(batch, 256, 4, 64, 64).permute(0, 2, 1, 3, 4).contiguous()

    def encode_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if x.ndim != 5 or tuple(x.shape[1:]) != (26, 10, 256, 256):
            raise ValueError(f"AgriFM expects [B,26,10,256,256], got {tuple(x.shape)}")
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 2))
        backbone_grad = any(parameter.requires_grad for parameter in self.backbone.parameters())
        context = torch.enable_grad() if backbone_grad and torch.is_grad_enabled() else torch.no_grad()
        with context:
            embedded = self.patch_embed(x)
            features = self.backbone(embedded)
        levels = features["features_list"]
        if len(levels) != 4:
            raise ValueError(f"AgriFM must expose patch embedding plus three Swin scales, got {len(levels)}")
        f1, f2 = levels[1], levels[2]
        f3 = features["encoder_features"]
        return f1, f2, f3

    def decode_features(self, f1: Tensor, f2: Tensor, f3: Tensor) -> Tensor:
        return self.decoder(f1, f2, f3)

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        f1, f2, f3 = self.encode_features(x)
        return self.decode_features(f1, f2, f3), self._f1_groups(f1)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)[0]


__all__ = ["AgriFMClassifier", "NativeFPN256", "load_pretrained"]
