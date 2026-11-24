"""Kenchi model package.

The heavy imports (torch) happen on first use, so the tokenizer and the chat
format can be imported without torch (the MLX runner does that).
"""

_LAZY = {
    "count_params": "kenchi.flops", "forward_macs_per_token": "kenchi.flops",
    "train_flops_per_token": "kenchi.flops",
    "A": "kenchi.mixers", "D": "kenchi.mixers", "L": "kenchi.mixers", "M": "kenchi.mixers",
    "RMSNorm": "kenchi.mixers", "round_to_64": "kenchi.mixers",
    "Kenchi": "kenchi.model", "KenchiConfig": "kenchi.model", "StubModel": "kenchi.model",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'kenchi' has no attribute {name!r}")
