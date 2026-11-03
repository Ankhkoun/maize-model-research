"""Verified CE3 -> D+GMM teacher -> student decoder initialization."""

import torch
from dgmm_moe.models.network import ClassifierModel, TeacherModel
from dgmm_moe.runtime import plan, sha
from dgmm_moe.io_utils import setup, state_sha
from dgmm_moe.checkpoints import restore_classifier, frozen_part


def teacher_initialized():
    setup(42)
    m = TeacherModel().cuda()
    cfg = plan()
    p = cfg["warmup_path"]
    if not cfg["warmup_checkpoint_sha256"] or not cfg["warmup_classifier_sha256"]:
        raise RuntimeError("Run the ordinary CE3 warmup stage before training the teacher")
    assert sha(p) == cfg["warmup_checkpoint_sha256"]
    st = torch.load(p, map_location="cpu", weights_only=False)
    assert st["initialization"] == "official_AgriFM_fresh_decoder_classifier_only_warmup"
    restore_classifier(m.classifier, st)
    assert state_sha(m.classifier.state_dict()) == cfg["warmup_classifier_sha256"]
    return m


def student_initialized():
    setup(42)
    m = ClassifierModel().cuda()
    cfg = plan()
    if not cfg["teacher_checkpoint_sha256"]:
        raise RuntimeError("A completed Dev-selected D+GMM teacher is required")
    assert sha(cfg["teacher_checkpoint"]) == cfg["teacher_checkpoint_sha256"]
    st = torch.load(cfg["teacher_checkpoint"], map_location="cpu", weights_only=False)
    assert st["epoch"] == cfg["teacher_selected_method_epoch"] and st["compact_model"]
    frozen = frozen_part(m.state_dict(), "classifier.")
    assert state_sha(frozen) == st["frozen_classifier_sha256"]
    decoder = {
        k[len("classifier.decoder.") :]: v
        for k, v in st["model"].items()
        if k.startswith("classifier.decoder.")
    }
    m.classifier.decoder.load_state_dict(decoder, strict=True)
    reconstructed = dict(m.state_dict())
    reconstructed.update({k: v for k, v in st["model"].items() if not k.startswith("classifier.")})
    assert state_sha(reconstructed) == st["model_full_state_sha256"]
    m.eval()
    return m
