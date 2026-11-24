"""src/kenchi/copy_probe.py — copy probe (run in the final eval when task == "lm").

Measures how well the model copies repeated context, without any training:

1. "random": rows of [prefix_len(=64) tokens of real validation text] + [s_1..s_L] +
   [s_1..s_L], s ~ U{vocab_size//8 .. vocab_size//2}; the metric is the top-1 accuracy on
   the second copy excluding its first 2 tokens -- the model's argmax at the
   positions predicting the second copy's tokens 3..L. n_rows(=64) rows per L.
2. "natural": a validation chunk of L tokens followed by the same chunk again
   (row = chunk + chunk); metrics are the accuracy on the second copy (the model's
   argmax at the positions predicting the second copy's tokens) and, for
   reference, the accuracy on the first copy (the model's argmax at the positions
   predicting the first copy's tokens). L in {64, 256}.

Result shape (matches the brief):

    {"random": {"16": acc_or_None, "64": .., "256": ..},
     "natural": {"64": {"first": .., "second": ..}, "256": {...}}}
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Optional

import torch

from kenchi.recall_task import recall_ids


def _acc(model, x, t_pos, t_tgt, m, autocast_ctx=None, row_chunk=16):
    """Mean over (b, e) with m[b, e] of [argmax(logits[b, t_pos[b,e]]) == t_tgt[b,e]].

    `x`: (N, T) rows; `t_pos`: (N, K) positions; `t_tgt`: (N, K) target tokens;
    `m`: (N, K) validity mask. Returns None if there are no valid positions.
    """
    N, K = t_pos.shape
    if not m.any():
        return None
    ok = torch.zeros(N, K, dtype=torch.bool, device=x.device)
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    with torch.no_grad():
        for b0 in range(0, N, row_chunk):
            b1 = min(N, b0 + row_chunk)
            xc = x[b0:b1]
            with ctx:
                logits = model(xc)["logits"]
            rows = torch.arange(b1 - b0, device=x.device).unsqueeze(1).expand(b1 - b0, K)
            pos = t_pos[b0:b1].clamp(0, xc.shape[1] - 1)
            pred = logits[rows, pos].argmax(-1)
            ok[b0:b1] = pred == t_tgt[b0:b1]
    return float(ok[m].float().mean().item())


def copy_probe_random(model, val_tokens, Ls=(16, 64, 256), n_rows=64,
                      prefix_len=64, device=None, autocast_ctx=None, seed=0,
                      vocab_size: int = 50304,
                      row_chunk: int = 16) -> Dict[str, Optional[float]]:
    """Random-token repeat probe; see the module docstring for the metric.

    Random tokens are drawn from the model's ordinary range [vocab_size//8,
    vocab_size//2).
    """
    dev = torch.device(device) if device is not None else val_tokens.device
    g = torch.Generator().manual_seed(int(seed))
    ids = recall_ids(vocab_size)
    lo, hi = ids.ordinary_lo, ids.ordinary_hi
    out: Dict[str, Optional[float]] = {}
    n_val = val_tokens.numel()
    for L in Ls:
        if L < 3 or prefix_len + 2 * L > n_val:
            out[str(L)] = None
            continue
        s = torch.randint(lo, hi, (n_rows, L), generator=g).to(dev)
        prefix = val_tokens[:prefix_len].to(dev).unsqueeze(0).expand(n_rows, prefix_len)
        row = torch.cat([prefix, s, s], dim=1)                     # (n_rows, prefix+2L)
        T = row.shape[1]
        base = prefix_len + L                                       # start of the 2nd copy
        # Accuracy on the second copy excluding its first 2 tokens: the model
        # predicts the second copy's tokens 3..L at positions base+1 .. base+L-2.
        ts = torch.arange(base + 1, base + L - 1, device=dev)
        if ts.numel() == 0:
            out[str(L)] = None
            continue
        ts = ts.view(1, -1).expand(n_rows, -1)
        tgt = row.gather(1, ts + 1)
        out[str(L)] = _acc(model, row, ts, tgt, torch.ones_like(ts, dtype=torch.bool),
                           autocast_ctx=autocast_ctx, row_chunk=row_chunk)
    return out


def copy_probe_natural(model, val_tokens, Ls=(64, 256), device=None,
                       autocast_ctx=None, seed=0, row_chunk: int = 16
                       ) -> Dict[str, Dict[str, Optional[float]]]:
    """Natural-text repeat probe; see the module docstring for the metrics."""
    dev = torch.device(device) if device is not None else val_tokens.device
    out: Dict[str, Dict[str, Optional[float]]] = {}
    n = val_tokens.numel()
    for L in Ls:
        if L < 2 or 2 * L > n:
            out[str(L)] = {"first": None, "second": None}
            continue
        g = torch.Generator().manual_seed(int(seed) + L)
        start = int(torch.randint(0, n - 2 * L + 1, (1,), generator=g).item())
        chunk = val_tokens[start:start + L].to(dev)
        if chunk.numel() < L:
            chunk = val_tokens[:L].to(dev)
        row = torch.cat([chunk, chunk], dim=0).unsqueeze(0)        # (1, 2L)
        # Second copy = tokens[L..2L-1]: its tokens are predicted at t = L-1..2L-2.
        ts2 = torch.arange(L - 1, 2 * L - 1, device=dev).unsqueeze(0)
        tgt2 = row[:, ts2 + 1]
        acc_second = _acc(model, row, ts2, tgt2,
                          torch.ones_like(ts2, dtype=torch.bool),
                          autocast_ctx=autocast_ctx, row_chunk=row_chunk)
        # First copy = tokens[0..L-1]: its tokens are predicted at t = 0..L-2.
        ts1 = torch.arange(0, L - 1, device=dev).unsqueeze(0)
        tgt1 = row[:, ts1 + 1]
        acc_first = _acc(model, row, ts1, tgt1,
                         torch.ones_like(ts1, dtype=torch.bool),
                         autocast_ctx=autocast_ctx, row_chunk=row_chunk)
        out[str(L)] = {"first": acc_first, "second": acc_second}
    return out
