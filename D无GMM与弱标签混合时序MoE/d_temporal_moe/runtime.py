"""Paths for one complete training flow; code/configuration fingerprints prevent drift."""

from pathlib import Path
import hashlib
import json
import os

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(
    os.environ.get("D_TEMPORAL_MOE_CONFIG", PACKAGE_ROOT / "configs/pure_moe_lambda0.json")
).resolve()
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
STAGE = os.environ.get("D_TEMPORAL_MOE_STAGE", "teacher")
if STAGE not in ("teacher", "student"):
    raise ValueError("Unknown training stage")
if CONFIG["seed"] != 42:
    raise ValueError("This version preserves the seed42 training schedule")



def mixing_weight():
    value = CONFIG["weak_label_weight"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0.0, 0.25):
        raise ValueError("Supported weak-label weights are 0.0 and 0.25")
    return float(value)


mixing_weight()

def path(name):
    value = Path(CONFIG["paths"][name]).expanduser()
    return value.resolve() if value.is_absolute() else (CONFIG_PATH.parent / value).resolve()


RUN_ROOT = path("run_dir")
TEACHER_ROOT = path("teacher_run_dir")
ROOT = TEACHER_ROOT if STAGE == "teacher" else RUN_ROOT / "student"
ASSETS = path("protocol_assets")
PRETRAIN = path("pretrained")
AGRIFM_SOURCE = path("agrifm_source")
EXPECTED = CONFIG["protocol_hashes"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_if_present(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}


def plan():
    teacher = TEACHER_ROOT
    warmup = read_if_present(teacher / "prepared/warmup_complete.json")
    completed = read_if_present(teacher / "checkpoints/completed.json")
    neighbors = read_if_present(teacher / "prepared/complete.json")
    return dict(
        source_pretrained_sha256=CONFIG["source_pretrained_sha256"],
        experts=dict(fit_seed=42030),
        warmup_path=str(teacher / "prepared/warmup.pt"),
        warmup_checkpoint_sha256=warmup.get("checkpoint_sha256"),
        warmup_classifier_sha256=warmup.get("classifier_sha256"),
        teacher_checkpoint=str(teacher / "checkpoints/best.pt"),
        teacher_checkpoint_sha256=completed.get("best_checkpoint_sha256"),
        teacher_selected_method_epoch=completed.get("best", {}).get("epoch"),
        parent_data_provenance=str(teacher / "prepared/data_provenance.json"),
        parent_data_provenance_sha256=neighbors.get("data_provenance_sha256"),
    )


def lock():
    result = {
        p.relative_to(PACKAGE_ROOT).as_posix(): sha(p)
        for p in sorted((PACKAGE_ROOT / "d_temporal_moe").rglob("*.py"))
    }
    upstream = PACKAGE_ROOT / "third_party/AgriFM/AgriFM/models/video_swin_transformer.py"
    result[upstream.relative_to(PACKAGE_ROOT).as_posix()] = sha(upstream)
    result["active_config_sha256"] = sha(CONFIG_PATH)
    return result


def require_model():
    if sha(PRETRAIN) != CONFIG["source_pretrained_sha256"]:
        raise ValueError("Pretrained checkpoint fingerprint mismatch")
    backbone = Path(AGRIFM_SOURCE) / "AgriFM/models/video_swin_transformer.py"
    if sha(backbone) != CONFIG["source_video_swin_sha256"]:
        raise ValueError("AgriFM backbone source fingerprint mismatch")


def require_inputs():
    require_model()
    for name in ("train_validation500.csv", "gtdev60.csv", "normalization_train1296.json"):
        if sha(Path(ASSETS) / name) != EXPECTED[name]:
            raise ValueError(f"Protocol fingerprint mismatch: {name}")
