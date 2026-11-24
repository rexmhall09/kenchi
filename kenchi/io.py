"""Load and save Kenchi weights in the release format (safetensors + config JSON)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from kenchi.model import Kenchi, KenchiConfig


def load_model(weights, config, device="cpu", dtype=None):
    """Build a Kenchi model from ``<run>_final.safetensors`` and its config JSON.

    ``config`` is a path to the JSON written by the trainer (the model config is
    under the key "model") or a dict.  Embeddings are tied: the file holds the
    token embedding once and ``lm_head.weight`` is the same tensor.
    Returns ``(model, model_config)``; the model is in eval mode.
    """
    from safetensors.torch import load_file

    cfg = config if isinstance(config, dict) else json.loads(Path(config).read_text())
    mcfg = KenchiConfig.from_dict(cfg.get("model", cfg))
    model = Kenchi(mcfg)
    sd = load_file(str(weights))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    allowed_missing = {"lm_head.weight"} if getattr(mcfg, "tie_embeddings", True) else set()
    bad = [k for k in missing if k not in allowed_missing]
    if bad or unexpected:
        raise RuntimeError(f"weights do not match the config: missing={bad} unexpected={list(unexpected)}")
    if "lm_head.weight" in missing and model.lm_head.weight is not model.tok_emb.weight:
        model.lm_head.weight = model.tok_emb.weight
    model.to(device)
    if dtype is not None:
        model.to(dtype)
    model.eval()
    return model, mcfg


def save_model(model, path, metadata=None):
    """Write fp32 safetensors, each tensor once (tied weights are not repeated)."""
    from safetensors.torch import save_file

    m = model
    while hasattr(m, "module") or hasattr(m, "_orig_mod"):
        m = getattr(m, "module", None) or getattr(m, "_orig_mod")
    out, seen = {}, set()
    for k, v in m.state_dict().items():
        if v.data_ptr() in seen:
            continue
        seen.add(v.data_ptr())
        out[k] = v.detach().to("cpu", torch.float32).contiguous().clone()
    meta = {"format": "pt"}
    meta.update({str(k): str(v) for k, v in (metadata or {}).items()})
    save_file(out, str(path), metadata=meta)
    return Path(path)


from kenchi.assistant import read_metadata  # noqa: E402,F401  (plain Python; the MLX runner uses it too)
