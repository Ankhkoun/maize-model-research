"""Compact checkpoints preserve learned tensors and verify the frozen backbone by hash."""

"Lossless checkpoints: shared frozen backbone by hash, learned tensors in-file."
from d_temporal_moe.runtime import PRETRAIN, plan
from d_temporal_moe.io_utils import state_sha


def frozen_part(state, prefix):
    return {
        k: v
        for k, v in state.items()
        if k.startswith(prefix) and (not k.startswith(prefix + "decoder."))
    }


def pack(payload):
    if not isinstance(payload, dict):
        return payload
    d = dict(payload)
    if "model" in d and isinstance(d["model"], dict):
        state = d["model"]
        frozen = frozen_part(state, "classifier.")
        d["model"] = {k: v for k, v in state.items() if k not in frozen}
        d["compact_model"] = True
        d["model_full_state_sha256"] = state_sha(state)
        d["frozen_classifier_sha256"] = state_sha(frozen)
    elif "classifier" in d and isinstance(d["classifier"], dict):
        state = d["classifier"]
        frozen = {k: v for k, v in state.items() if not k.startswith("decoder.")}
        d["classifier"] = {k: v for k, v in state.items() if k not in frozen}
        d["compact_classifier"] = True
        d["classifier_full_state_sha256"] = state_sha(state)
        d["frozen_classifier_sha256"] = state_sha(frozen)
    else:
        return payload
    d["frozen_reference"] = dict(
        path=PRETRAIN,
        sha256=plan()["source_pretrained_sha256"],
        format="official_AgriFM_frozen_backbone_reference_v1",
    )
    return d


def restore_model(m, payload):
    if not payload.get("compact_model"):
        m.load_state_dict(payload["model"], strict=True)
        return
    state = m.state_dict()
    frozen = frozen_part(state, "classifier.")
    assert state_sha(frozen) == payload["frozen_classifier_sha256"]
    result = m.load_state_dict(payload["model"], strict=False)
    assert not result.unexpected_keys and set(result.missing_keys) == set(frozen)
    assert state_sha(m.state_dict()) == payload["model_full_state_sha256"]


def restore_classifier(m, payload):
    if not payload.get("compact_classifier"):
        m.load_state_dict(payload["classifier"], strict=True)
        return
    frozen = {k: v for k, v in m.state_dict().items() if not k.startswith("decoder.")}
    assert state_sha(frozen) == payload["frozen_classifier_sha256"]
    result = m.load_state_dict(payload["classifier"], strict=False)
    assert not result.unexpected_keys and set(result.missing_keys) == set(frozen)
    assert state_sha(m.state_dict()) == payload["classifier_full_state_sha256"]
