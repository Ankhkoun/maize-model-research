"""Regressions for the two requested supervision configurations."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from d_temporal_moe.models.temporal_moe import soft_target

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("weight", [0.0, 0.25])
def test_anchor_and_student_mixing(weight):
    labels = torch.tensor([0, 1, 255])
    teacher = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
    posterior = torch.tensor([0.9, 0.1, 0.8], requires_grad=True)
    anchor = soft_target(weight, labels, teacher)
    target = soft_target(weight, labels, posterior, teacher, torch.tensor([True, False, True]))
    assert torch.equal(anchor, weight * (labels == 1) + (1 - weight) * teacher.detach())
    expected = weight * (labels == 1) + (1 - weight) * torch.tensor([0.9, 0.2, 0.8])
    assert torch.equal(target, expected)
    assert not anchor.requires_grad and not target.requires_grad


def test_both_configs_share_teacher_but_isolate_students():
    configs = [json.loads((ROOT / "configs" / name).read_text()) for name in
               ["pure_moe_lambda0.json", "weak_mix_lambda025.json"]]
    assert [c["weak_label_weight"] for c in configs] == [0.0, 0.25]
    assert configs[0]["paths"]["teacher_run_dir"] == configs[1]["paths"]["teacher_run_dir"]
    assert configs[0]["paths"]["run_dir"] != configs[1]["paths"]["run_dir"]
    for config, expected in zip(configs, [0.0, 0.25]):
        name = "pure_moe_lambda0.json" if expected == 0 else "weak_mix_lambda025.json"
        env = dict(os.environ, D_TEMPORAL_MOE_CONFIG=str(ROOT / "configs" / name),
                   D_TEMPORAL_MOE_STAGE="student", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-B", "-c",
            "from pathlib import Path; from d_temporal_moe.runtime import mixing_weight,plan; "
            f"assert mixing_weight()=={expected}; assert Path(plan()['teacher_checkpoint']).parts[-3:]==('shared_d_teacher','checkpoints','best.pt')"],
            cwd=ROOT, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_package_contains_no_loss_gmm_imports():
    for file in (ROOT / "d_temporal_moe").rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert "gmm" not in (node.module or "").lower()
                assert "sklearn.mixture" not in (node.module or "")
            if isinstance(node, ast.Import):
                assert all("gmm" not in alias.name.lower() and "sklearn.mixture" not in alias.name for alias in node.names)
