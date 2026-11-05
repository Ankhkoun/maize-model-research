"""Checks across refactored module boundaries; no training data or GPU required."""

from pathlib import Path
import json
import subprocess
import sys
import torch
from torch import nn
from d_temporal_moe.models.classifier import NativeFPN256
from d_temporal_moe.models.noise import Noise
from d_temporal_moe.checkpoints import pack, restore_model
from d_temporal_moe.io_utils import state_sha, save
from d_temporal_moe.data import transform
from d_temporal_moe.training.teacher import joint

ROOT = Path(__file__).resolve().parents[1]


def test_fpn_output_and_parameter_count():
    torch.set_num_threads(1)
    decoder = NativeFPN256()
    assert sum((p.numel() for p in decoder.parameters())) == 689666
    features = [torch.randn(1, 1024, s, s) for s in (8, 4, 2)]
    logits = decoder(*features)
    assert logits.shape == (1, 2, 256, 256)
    logits.square().mean().backward()
    assert all((p.grad is not None and p.grad.isfinite().all() for p in decoder.parameters()))


class Toy(nn.Module):

    def __init__(self):
        super().__init__()
        self.classifier = nn.Module()
        self.classifier.backbone = nn.Linear(3, 3)
        self.classifier.backbone.requires_grad_(False)
        self.classifier.decoder = nn.Linear(3, 2)
        self.module = Noise()


def test_compact_checkpoint_round_trip(tmp_path):
    torch.manual_seed(42)
    model = Toy()
    original = state_sha(model.state_dict())
    payload = pack(
        dict(model={k: v.detach().clone() for k, v in model.state_dict().items()}, epoch=3)
    )
    assert payload["compact_model"]
    assert all((not k.startswith("classifier.backbone.") for k in payload["model"]))
    with torch.no_grad():
        model.classifier.decoder.weight.add_(1)
    restore_model(model, payload)
    assert state_sha(model.state_dict()) == original
    save(tmp_path / "check.pt", dict(model=model.state_dict(), epoch=3))
    restored = torch.load(tmp_path / "check.pt", map_location="cpu", weights_only=False)
    assert restored["model_full_state_sha256"] == original


class TinyTeacher(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = Noise()
        self.logits = nn.Parameter(torch.randn(1, 2, 256, 256))

    def forward(self, b):
        return dict(logits=self.logits)


def test_teacher_joint_alignment_and_gradient(monkeypatch):
    torch.set_num_threads(1)
    from contextlib import nullcontext

    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    model = TinyTeacher()
    label = torch.randint(0, 2, (1, 256, 256))
    label[:, 0, 0] = 255
    entry = dict(labels=torch.randint(0, 2, (2, 64, 64)), weights=torch.rand(3, 3, 64, 64))
    entry["weights"] /= entry["weights"].sum(1, keepdim=True)
    trans = (1, True, False)
    b = dict(target=transform(label, trans), valid=transform(label != 255, trans), transform=trans)
    loss, posterior = joint(model, b, entry)
    assert loss.isfinite() and posterior.shape == (1, 2, 256, 256)
    loss.backward()
    assert model.module.raw.grad.isfinite().all() and model.module.raw.grad.abs().sum() > 0
    assert model.logits.grad.abs().sum() > 0
    assert (model.logits.grad[:, :, 255, 0] == 0).all() == (b["target"][0, 255, 0] == 255)



def test_cli_inspection_does_not_create_outputs(tmp_path):
    config = json.loads((ROOT / "configs/pure_moe_lambda0.json").read_text())
    output = tmp_path / "must_remain_absent"
    config["paths"]["run_dir"] = str(output)
    c = tmp_path / "config.json"
    c.write_text(json.dumps(config), encoding="utf-8")
    for action in ("stages", "show-config"):
        r = subprocess.run(
            [sys.executable, "-m", "d_temporal_moe", "--config", str(c), action],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
    assert not output.exists()


def test_classifier_freeze_and_training_mode(monkeypatch):
    from d_temporal_moe.models import network

    class BackboneAdapter(nn.Module):

        def __init__(self, *args):
            super().__init__()
            self.backbone = nn.Conv2d(3, 4, 1)
            self.decoder = nn.Conv2d(4, 2, 1)

        def encode_features(self, x):
            feature = self.backbone(x)
            return (feature, feature, feature)

        def decode_features(self, a, b, c):
            return self.decoder(a)

    monkeypatch.setattr(network, "AgriFMClassifier", BackboneAdapter)
    model = network.TeacherModel()
    model.train_mode()
    assert not model.classifier.backbone.training and model.classifier.decoder.training
    assert model.module.training
    before = state_sha(model.classifier.backbone.state_dict())
    logits = model(dict(agrifm_cube=torch.randn(1, 3, 4, 4)))["logits"]
    optimizer = torch.optim.AdamW(model.classifier.decoder.parameters(), lr=0.00015)
    logits.square().mean().backward()
    optimizer.step()
    assert before == state_sha(model.classifier.backbone.state_dict())
    assert not any((p.grad is not None for p in model.classifier.backbone.parameters()))
    assert any((p.grad is not None for p in model.classifier.decoder.parameters()))


def test_all_pipeline_entrypoints_import():
    import importlib
    from d_temporal_moe.__main__ import ACTIONS

    for stage, module, function in ACTIONS.values():
        assert callable(getattr(importlib.import_module(module), function))


def test_data_and_observation_interfaces(tmp_path, monkeypatch):
    import numpy as np
    from d_temporal_moe.data import Assets, transform
    from d_temporal_moe.observations import read

    assets = object.__new__(Assets)
    assets.cubes = tmp_path / "cubes"
    assets.labels = tmp_path / "labels"
    assets.label_hashes = {}
    assets.mean = np.full((1, 10, 1, 1), 0.1, dtype=np.float32)
    assets.std = np.full((1, 10, 1, 1), 0.2, dtype=np.float32)
    sid = "synthetic"
    folder = assets.cubes / sid
    folder.mkdir(parents=True)
    meta = dict(
        sample_id=sid,
        crs="EPSG:3857",
        band_names=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"],
    )
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    quality = [dict(slot_id=f"t{i+1:02d}") for i in range(26)]
    (folder / "time_quality.json").write_text(json.dumps(quality), encoding="utf-8")
    raw = np.full((26, 10, 256, 256), 0.2, dtype=np.float32)
    source = np.full((26, 256, 256), 2, dtype=np.uint8)
    source[0] = 1
    labels = np.zeros((256, 256), dtype=np.uint8)
    labels[0, 1] = 1
    labels[0, 0] = 255
    loads = []

    def fake_load(path, **kwargs):
        name = Path(path).name
        loads.append(name)
        return {"x_cube.npy": raw, "source_map.npy": source, "pseudo_label.npy": labels}[name]

    monkeypatch.setattr(np, "load", fake_load)
    row = dict(sample_id=sid, region_id="synthetic-region")
    t = (1, True, False)
    sample = assets.load(row, t=t)
    assert sample["agrifm_cube"].shape == (26, 10, 256, 256)
    assert torch.equal(sample["target"], transform(torch.from_numpy(labels.astype(np.int64)), t))
    assert torch.equal(sample["valid"], sample["target"] != 255)
    loads.clear()
    no_target = assets.load(row, need_label=False)
    assert "target" not in no_target and "pseudo_label.npy" not in loads
    x, observed, days = read(assets, row)
    assert x.shape == (65536, 26, 10) and observed.shape == (65536, 26)
    assert observed[:, 0].all() and not observed[:, 1:].any()
    assert torch.allclose(x[:, 0], torch.full_like(x[:, 0], 0.5))
    assert not x[:, 1:].any() and days[0] == 94 and days[-1] == 269.5


def test_warmup_ordinary_ce_and_gradient():
    from d_temporal_moe.training.losses import ordinary_ce

    torch.manual_seed(42)
    logits = torch.randn(1, 2, 3, 4, requires_grad=True)
    target = torch.randint(0, 2, (1, 3, 4))
    target[0, 0, 0] = 255
    b = dict(target=target, valid=target != 255)
    loss = ordinary_ce(logits, b)
    expected = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert not logits.grad[0, :, 0, 0].any()


def test_standalone_prediction_help_needs_no_training_artifacts():
    result = subprocess.run(
        [sys.executable, "-m", "d_temporal_moe.predict", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout and "--cube" in result.stdout


def test_neighbor_preparation_uses_configured_pretrain(tmp_path, monkeypatch):
    import pytest
    from d_temporal_moe.training import neighbors, teacher

    checkpoint = tmp_path / "configured_pretrain.pt"
    checkpoint.write_bytes(b"synthetic checkpoint fingerprint")
    expected = neighbors.sha(checkpoint)
    monkeypatch.setattr(neighbors, "ROOT", tmp_path)
    monkeypatch.setattr(neighbors, "PRETRAIN", checkpoint)
    monkeypatch.setattr(neighbors, "setup", lambda seed: None)
    monkeypatch.setattr(neighbors, "lock", lambda: {})
    monkeypatch.setattr(neighbors, "Assets", lambda: object())
    monkeypatch.setattr(neighbors, "splits", lambda: ([], []))
    monkeypatch.setattr(neighbors, "plan", lambda: dict(source_pretrained_sha256=expected))
    monkeypatch.setattr(teacher, "initialized", lambda: nn.Identity())

    class ReachedFeatureProjection(Exception):
        pass

    def stop_before_cuda(*args, **kwargs):
        raise ReachedFeatureProjection

    monkeypatch.setattr(torch, "randn", stop_before_cuda)
    with pytest.raises(ReachedFeatureProjection):
        neighbors.prepare()
