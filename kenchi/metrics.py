"""Metrics for Phase 1 (src/kenchi/metrics.py).

ECE, Brier, AUROC (no sklearn at run time), the entity-like target mask (built once
with tiktoken's gpt2 encoding and cached under src/kenchi/assets/ so training jobs
never need tiktoken), the per-position signal features on validation (next-token
correctness / confidence / entropy / uncertainty-head / JEPA / loop signals), and the
uncertainty-signal table of docs/PHASE1_DESIGN.md §6.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
ENTITY_MASK_CACHE = ASSETS_DIR / "entity_mask.npy"


def _np(x) -> np.ndarray:
    """Tensor/array -> CPU numpy array (GPU-safe: detach + cpu first)."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


# ---------------------------------------------------------------- calibration


def ece(confidences, correct, n_bins: int = 15) -> float:
    """Expected calibration error: count-weighted mean of |bin accuracy - bin confidence|.

    15 equal-width bins over [0, 1]; the last bin is inclusive of 1.0. Empty bins
    contribute nothing. NaN if there are no positions.
    """
    conf = _np(confidences).astype(np.float64).reshape(-1)
    corr = _np(correct).astype(np.float64).reshape(-1)
    n = conf.shape[0]
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        if i < n_bins - 1:
            m = (conf >= edges[i]) & (conf < edges[i + 1])
        else:
            m = (conf >= edges[i]) & (conf <= edges[i + 1])
        c = int(m.sum())
        if c:
            err += c * abs(float(corr[m].mean()) - float(conf[m].mean()))
    return err / n


def brier(confidences, correct) -> float:
    """Mean squared error of the confidence against the correctness indicator."""
    conf = _np(confidences).astype(np.float64).reshape(-1)
    corr = _np(correct).astype(np.float64).reshape(-1)
    if conf.shape[0] == 0:
        return float("nan")
    return float(np.mean((conf - corr) ** 2))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.shape[0]:
        j = i
        while j + 1 < x.shape[0] and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def auroc(scores, labels) -> Optional[float]:
    """Mann-Whitney AUROC with average ranks for ties; None when degenerate
    (no positives, no negatives, or empty input)."""
    s = _np(scores).astype(np.float64).reshape(-1)
    y = _np(labels).astype(np.int64).reshape(-1)
    if s.shape[0] == 0:
        return None
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = _average_ranks(s)
    auc = (float(ranks[y == 1].sum()) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return float(auc)


def dist_entropy(p: torch.Tensor) -> torch.Tensor:
    """Token-level entropy H(p) = -sum p log p over the last dim (probabilities)."""
    p = p.clamp_min(1e-30)
    return -(p * torch.log(p)).sum(-1)


# ---------------------------------------------------------------- entity mask


def _build_entity_mask() -> np.ndarray:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    n = int(enc.n_vocab)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        s = enc.decode([i])
        if len(s) >= 2 and s[0] == " " and (s[1].isupper() or s[1].isdigit()):
            mask[i] = True
    return mask


def _gpt2_entity_mask() -> np.ndarray:
    if ENTITY_MASK_CACHE.exists():
        return np.load(ENTITY_MASK_CACHE)
    m = _build_entity_mask()
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ENTITY_MASK_CACHE, m)
    return m


def entity_mask(vocab_size: int = 50304, path=None) -> Optional[np.ndarray]:
    """Boolean mask over `vocab_size` tokens marking "entity-like" targets.

    With `path`: load the mask from that file (e.g. the reduced-vocab mask written
    by scripts/make_small_vocab_shards.py, length = the model vocab). Otherwise
    use the packaged GPT-2 mask (built once with tiktoken's gpt2 encoding and
    cached as src/kenchi/assets/entity_mask.npy): pad with False when
    vocab_size > len, and return None when vocab_size < len (a small vocab
    without a provided mask -- callers skip the entity-like metrics with null).
    """
    if path is not None:
        return np.load(Path(path).expanduser()).astype(np.bool_)
    m = _gpt2_entity_mask()
    if vocab_size > m.shape[0]:
        padded = np.zeros(vocab_size, dtype=np.bool_)
        padded[: m.shape[0]] = m
        return padded
    if vocab_size < m.shape[0]:
        return None
    return m


# ---------------------------------------------------------------- validation


def val_loss(model, tokens, chunk: int = 16, n_loops=None, cfg=None,
             autocast_ctx=None) -> float:
    """Mean next-token NLL (nats/token) of `model` on the 1-D token stream `tokens`.

    The stream is cut into non-overlapping windows of T+1 tokens with stride T
    (T = cfg.seq_len when cfg is given, else 4096): every position t in [0, n - 2]
    is scored exactly once as CE(logits[t], tokens[t + 1]) -- the same convention as
    the training loader (windows of T+1 tokens; the model scores the first T
    positions). Windows are forwarded in batches of `chunk` rows under
    `autocast_ctx`; logits are cast to float32 before the CE.
    """
    T = int(cfg.seq_len) if cfg is not None else 4096
    n = tokens.numel()
    if n < 2:
        return float("nan")
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    tot = 0.0
    cnt = 0
    with torch.no_grad():
        if n < T + 1:
            x = tokens.unsqueeze(0)
            with ctx:
                logits = model(x, n_loops=n_loops)["logits"]
            lp = logits[0, : n - 1].float()
            tgt = tokens[1:n]
            tot += float(F.cross_entropy(lp, tgt, reduction="sum"))
            cnt += tgt.numel()
        else:
            starts = torch.arange(0, n - T, T, device=tokens.device)
            for s0 in range(0, starts.numel(), chunk):
                s = starts[s0:s0 + chunk]
                x = tokens[s.unsqueeze(1) + torch.arange(T + 1, device=tokens.device).unsqueeze(0)]
                with ctx:
                    logits = model(x, n_loops=n_loops)["logits"]
                lp = logits[:, :-1].float()
                tgt = x[:, 1:]
                tot += float(F.cross_entropy(lp.reshape(-1, lp.shape[-1]),
                                              tgt.reshape(-1), reduction="sum"))
                cnt += tgt.numel()
    return tot / max(cnt, 1)


@dataclass
class ValFeatures:
    """Per-position model features on validation tokens (teacher-forced).

    Array index t is the model's prediction of tokens[t + 1]; all arrays have
    length n - 1 where n = tokens.numel(). Optional fields are None when the
    model (or its config) has no such output; `jepa_err` may contain NaN where
    the JEPA error is not defined.
    """
    correct: torch.Tensor
    conf: torch.Tensor
    entropy: torch.Tensor
    u_tok: Optional[torch.Tensor] = None
    u_span: Optional[torch.Tensor] = None
    jepa_logvar: Optional[torch.Tensor] = None
    jepa_err: Optional[torch.Tensor] = None
    loop_delta_last: Optional[torch.Tensor] = None


def val_features(model, tokens, chunk: int = 16, cfg=None,
                 autocast_ctx=None) -> ValFeatures:
    """Compute `ValFeatures` on the 1-D token stream `tokens`.

    Position t is the model's teacher-forced prediction of tokens[t + 1]. The
    stream is cut into non-overlapping windows of T+1 tokens with stride T
    (T = cfg.seq_len when cfg is given, else 4096) and forwarded in batches of
    `chunk` rows under `autocast_ctx`; logits are cast to float32 before the
    softmax/argmax.
    """
    T = int(cfg.seq_len) if cfg is not None else 4096
    n = tokens.numel()
    P = max(0, n - 1)
    dev = tokens.device
    feat = ValFeatures(
        correct=torch.zeros(P, dtype=torch.bool, device=dev),
        conf=torch.zeros(P, dtype=torch.float32, device=dev),
        entropy=torch.zeros(P, dtype=torch.float32, device=dev),
    )
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    u_tok = u_span = jepa_logvar = jepa_err = loop_delta_last = None
    with torch.no_grad():
        if n < T + 1:
            x = tokens.unsqueeze(0)
            with ctx:
                out = model(x, return_aux=True)
            logits = out["logits"][0, : n - 1].float()
            p = torch.softmax(logits, dim=-1)
            pred = p.argmax(-1)
            pos = torch.arange(0, n - 1, device=dev)
            feat.correct[pos] = pred == tokens[1:n]
            feat.conf[pos] = p.amax(-1)
            feat.entropy[pos] = dist_entropy(p)
            if "u_tok_logit" in out and u_tok is None:
                u_tok = torch.zeros(P, dtype=torch.float32, device=dev)
            if u_tok is not None:
                u_tok[pos] = torch.sigmoid(out["u_tok_logit"][0, : n - 1].float())
            if "u_span_logit" in out and u_span is None:
                u_span = torch.zeros(P, dtype=torch.float32, device=dev)
            if u_span is not None:
                u_span[pos] = torch.sigmoid(out["u_span_logit"][0, : n - 1].float())
            if "jepa_logvar" in out and jepa_logvar is None:
                jepa_logvar = torch.zeros(P, dtype=torch.float32, device=dev)
            if jepa_logvar is not None:
                jepa_logvar[pos] = out["jepa_logvar"][0, : n - 1].float()
            if "jepa_err" in out and jepa_err is None:
                jepa_err = torch.full((P,), float("nan"), device=dev, dtype=torch.float32)
            if jepa_err is not None:
                jepa_err[pos] = out["jepa_err"][0, : n - 1].float()
            if "loop_delta" in out and loop_delta_last is None:
                loop_delta_last = torch.zeros(P, dtype=torch.float32, device=dev)
            if loop_delta_last is not None:
                loop_delta_last[pos] = out["loop_delta"][0, : n - 1, -1].float()
        else:
            starts = torch.arange(0, n - T, T, device=dev)
            for s0 in range(0, starts.numel(), chunk):
                s = starts[s0:s0 + chunk]
                x = tokens[s.unsqueeze(1) + torch.arange(T + 1, device=dev).unsqueeze(0)]
                with ctx:
                    out = model(x, return_aux=True)
                logits = out["logits"][:, :-1].float()
                p = torch.softmax(logits, dim=-1)
                pred = p.argmax(-1)
                pos = s.unsqueeze(1) + torch.arange(T, device=dev).unsqueeze(0)  # (c, T)
                feat.correct[pos] = pred == x[:, 1:]
                feat.conf[pos] = p.amax(-1)
                feat.entropy[pos] = dist_entropy(p)
                if "u_tok_logit" in out and u_tok is None:
                    u_tok = torch.zeros(P, dtype=torch.float32, device=dev)
                if u_tok is not None:
                    u_tok[pos] = torch.sigmoid(out["u_tok_logit"][:, :-1].float())
                if "u_span_logit" in out and u_span is None:
                    u_span = torch.zeros(P, dtype=torch.float32, device=dev)
                if u_span is not None:
                    u_span[pos] = torch.sigmoid(out["u_span_logit"][:, :-1].float())
                if "jepa_logvar" in out and jepa_logvar is None:
                    jepa_logvar = torch.zeros(P, dtype=torch.float32, device=dev)
                if jepa_logvar is not None:
                    jepa_logvar[pos] = out["jepa_logvar"][:, :-1].float()
                if "jepa_err" in out and jepa_err is None:
                    jepa_err = torch.full((P,), float("nan"), device=dev, dtype=torch.float32)
                if jepa_err is not None:
                    jepa_err[pos] = out["jepa_err"][:, :-1].float()
                if "loop_delta" in out and loop_delta_last is None:
                    loop_delta_last = torch.zeros(P, dtype=torch.float32, device=dev)
                if loop_delta_last is not None:
                    loop_delta_last[pos] = out["loop_delta"][:, :-1, -1].float()
    feat.u_tok = u_tok
    feat.u_span = u_span
    feat.jepa_logvar = jepa_logvar
    feat.jepa_err = jepa_err
    feat.loop_delta_last = loop_delta_last
    return feat


def next8_labels(correct: torch.Tensor) -> torch.Tensor:
    """Label[t]: True when at least 4 of the next 8 next-token predictions
    (t+1..t+8) are correct ("next-8-tokens mostly correct", >= 50%).

    Index t of `correct` is the prediction of token t+1. Positions where fewer
    than 8 predictions exist (t > n - 10) are labelled False.
    """
    P = correct.numel()
    lab = torch.zeros(P, dtype=torch.bool, device=correct.device)
    if P < 8:
        return lab
    tot = torch.zeros(P, dtype=torch.float32, device=correct.device)
    t_idx = torch.arange(P, device=correct.device)
    for k in range(8):
        seg = torch.zeros(P, dtype=torch.float32, device=correct.device)
        m = t_idx + k <= P - 1
        if m.any():
            seg[m] = correct[t_idx[m] + k].to(torch.float32)
        tot += seg
    lab = tot >= 4
    lab[t_idx > P - 8] = False
    return lab


def _argmax_vec(model, tokens, n_loops=None, chunk: int = 16, cfg=None,
                autocast_ctx=None) -> torch.Tensor:
    """argmax of the model's logits at every position t in [0, n - 2].

    The stream is cut into non-overlapping windows of T+1 tokens with stride T
    (T = cfg.seq_len when cfg is given, else 4096) and forwarded in batches of
    `chunk` rows under `autocast_ctx`.
    """
    T = int(cfg.seq_len) if cfg is not None else 4096
    n = tokens.numel()
    P = max(0, n - 1)
    out = torch.zeros(P, dtype=torch.long, device=tokens.device)
    if P == 0:
        return out
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    with torch.no_grad():
        if n < T + 1:
            x = tokens.unsqueeze(0)
            with ctx:
                logits = model(x, n_loops=n_loops)["logits"]
            out[: n - 1] = logits[0, : n - 1].argmax(-1)
        else:
            starts = torch.arange(0, n - T, T, device=tokens.device)
            for s0 in range(0, starts.numel(), chunk):
                s = starts[s0:s0 + chunk]
                x = tokens[s.unsqueeze(1) + torch.arange(T + 1, device=tokens.device).unsqueeze(0)]
                with ctx:
                    logits = model(x, n_loops=n_loops)["logits"]
                pos = s.unsqueeze(1) + torch.arange(T, device=tokens.device).unsqueeze(0)
                out[pos] = logits[:, :-1].argmax(-1)
    return out


def halting_depth(model, tokens, n_max: int, chunk: int = 16, cfg=None,
                  autocast_ctx=None) -> torch.Tensor:
    """Per-position halting depth for looped configs (docs/PHASE1_DESIGN.md §6).

    Runs the eval token sequence with n_loops = 1..n_max; the depth at position t
    is the smallest n whose argmax equals the argmax at n_max (at n_max itself, so
    every position gets depth <= n_max). Returns a long tensor of shape (n - 1,).
    """
    n_max = max(1, int(n_max))
    P = max(0, tokens.numel() - 1)
    if P == 0:
        return torch.zeros(0, dtype=torch.long, device=tokens.device)
    depths = torch.full((P,), n_max + 1, dtype=torch.long, device=tokens.device)
    argmax_N = _argmax_vec(model, tokens, n_loops=n_max, chunk=chunk, cfg=cfg,
                           autocast_ctx=autocast_ctx)
    for k in range(1, n_max + 1):
        argmax_k = _argmax_vec(model, tokens, n_loops=k, chunk=chunk, cfg=cfg,
                               autocast_ctx=autocast_ctx)
        match = (argmax_k == argmax_N) & (depths == n_max + 1)
        depths[match] = k
    return depths


# ---------------------------------------------------------------- uncertainty table


def logistic_auroc(features, labels, steps: int = 200, lr: float = 0.5,
                   seed: int = 0) -> Optional[float]:
    """AUROC of a tiny torch logistic regression on `features` (P, k) -> labels.

    Fit on a held-out half of the rows (the first half), AUROC evaluated on the
    other half. Uses LBFGS in torch. Returns None when the data is degenerate.
    """
    f = _np(features).astype(np.float32)
    y = _np(labels).astype(np.int64)
    if f.ndim != 2 or f.shape[0] < 6:
        return None
    n = f.shape[0]
    half = n // 2
    if half == 0 or n - half == 0:
        return None
    # held-out half: seeded permutation so each half has both classes in general
    perm = np.random.default_rng(seed).permutation(n)
    tr, te = perm[:half], perm[half:]
    ftr, ytr = f[tr], y[tr]
    fte, yte = f[te], y[te]
    if ytr.sum() == 0 or (ytr == 0).all():
        return None
    if yte.sum() == 0 or (yte == 0).all():
        return None
    lin = torch.nn.Linear(f.shape[1], 1)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.zero_()
    x = torch.tensor(ftr)
    yt = torch.tensor(ytr, dtype=torch.float32)
    opt = torch.optim.LBFGS(lin.parameters(), lr=lr, max_iter=60,
                             line_search_fn="strong_wolfe", tolerance_grad=1e-7,
                             tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(lin(x).squeeze(-1))
        loss = F.binary_cross_entropy(p, yt)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        p = torch.sigmoid(lin(torch.tensor(fte)).squeeze(-1))
    return auroc(p.numpy(), yte)


def _auroc_t(sig, labels) -> Optional[float]:
    if sig is None or labels is None or sig.numel() == 0:
        return None
    if labels.numel() != sig.numel():
        return None
    return auroc(sig, labels)


def _orient(raw: Optional[float]) -> Optional[float]:
    """Orientation-normalised AUROC: max(raw, 1 - raw) -- signals have different
    directions (entropy up = wrong, max-prob up = right)."""
    if raw is None:
        return None
    return max(raw, 1.0 - raw)


def _lr_auroc(ent, labels, sig=None) -> Optional[float]:
    """AUROC of a tiny logistic regression on [entropy] or [entropy, signal].

    Fit on a held-out half of the positions (`logistic_auroc`); direction-free by
    construction (the LR fits the label probability).
    """
    if ent is None or labels is None:
        return None
    if ent.numel() == 0 or labels.numel() != ent.numel():
        return None
    parts = [_np(ent).astype(np.float32)]
    if sig is not None:
        if sig.numel() != ent.numel():
            return None
        parts.append(_np(sig).astype(np.float32))
    f = np.stack(parts, axis=1)
    return logistic_auroc(f, labels)


# The uncertainty signals of docs/PHASE1_DESIGN.md §6, in table order.
SIGNAL_NAMES = ("max_prob", "entropy", "u_tok", "u_span", "jepa_logvar", "jepa_err",
                "loop_delta_last", "loop_halt")


def uncertainty_table(val_signals: Dict[str, torch.Tensor],
                      val_correct: torch.Tensor,
                      val_next8: Optional[torch.Tensor],
                      val_entropy: torch.Tensor,
                      recall_labels: Optional[torch.Tensor],
                      recall_signals: Optional[Dict[str, torch.Tensor]],
                      recall_entropy_at_A: Optional[torch.Tensor]) -> Dict:
    """Build the §6 uncertainty-signal table.

    For every signal present in `val_signals`, for each target t in
    {next_token, next8, recall}:
      - `auroc_raw_<t>`: AUROC of the signal vs the target label as-is;
      - `auroc_<t>`: orientation-normalised `max(raw, 1 - raw)` -- signals have
        different directions (entropy up = wrong, max-prob up = right);
      - `lr_auroc_<t>`: AUROC of a logistic regression on [entropy, signal]
        (held-out half, `logistic_auroc`);
      - `lr_auroc_entropy_only_<t>`: same logistic AUROC with entropy alone, so
        the increment from the signal is visible.
    Targets: (i) next-token correctness on validation, (ii) "next-8-tokens mostly
    correct" (>= 4 of the next 8 correct, per `next8_labels`), (iii) recall answer
    exact-match (signal read at `<A>` of the recall docs, from `recall_signals`).
    Signals the config does not have are reported as null.

    - `val_signals`: {signal name: per-position array on validation}.
    - `val_correct`: per-position next-token correctness on validation.
    - `val_next8`: per-position next-8-mostly-correct labels (from `next8_labels`).
    - `val_entropy`: per-position token entropy on validation.
    - `recall_labels`: per-query recall answer exact-match (from the recall evaluator).
    - `recall_signals`: {signal name: per-query signal read at `<A>`}.
    - `recall_entropy_at_A`: per-query entropy read at `<A>` (the entropy feature
      of the recall LRs).
    """
    table: Dict = {}
    P = int(val_correct.numel())
    for name in SIGNAL_NAMES:
        sig = val_signals.get(name)
        if sig is None:
            table[name] = None
            continue
        sig = sig[:P]
        m = ~torch.isnan(sig)
        entry: Dict[str, Optional[float]] = {}
        # (i) next-token correctness on validation
        entry["auroc_raw_next_token"] = _auroc_t(sig[m], val_correct[m])
        entry["auroc_next_token"] = _orient(entry["auroc_raw_next_token"])
        # (ii) next-8-mostly-correct on validation
        if val_next8 is not None:
            entry["auroc_raw_next8"] = _auroc_t(sig[m], val_next8[m])
            entry["auroc_next8"] = _orient(entry["auroc_raw_next8"])
        else:
            entry["auroc_raw_next8"] = None
            entry["auroc_next8"] = None
        # (iii) recall answer exact-match, signal read at `<A>`
        rec_ok = (recall_labels is not None and recall_signals is not None
                  and name in recall_signals and recall_entropy_at_A is not None)
        if rec_ok:
            rs = recall_signals[name]
            mr = ~torch.isnan(rs)
            entry["auroc_raw_recall"] = _auroc_t(rs[mr], recall_labels[mr])
            entry["auroc_recall"] = _orient(entry["auroc_raw_recall"])
        else:
            entry["auroc_raw_recall"] = None
            entry["auroc_recall"] = None
        # logistic regressions on [entropy, signal] and entropy alone, per target
        entry["lr_auroc_next_token"] = _lr_auroc(val_entropy[m], val_correct[m], sig[m])
        entry["lr_auroc_entropy_only_next_token"] = _lr_auroc(val_entropy[m], val_correct[m])
        if val_next8 is not None:
            entry["lr_auroc_next8"] = _lr_auroc(val_entropy[m], val_next8[m], sig[m])
            entry["lr_auroc_entropy_only_next8"] = _lr_auroc(val_entropy[m], val_next8[m])
        else:
            entry["lr_auroc_next8"] = None
            entry["lr_auroc_entropy_only_next8"] = None
        if rec_ok:
            mr = ~torch.isnan(recall_signals[name])
            entry["lr_auroc_recall"] = _lr_auroc(recall_entropy_at_A[mr],
                                                 recall_labels[mr],
                                                 recall_signals[name][mr])
            entry["lr_auroc_entropy_only_recall"] = _lr_auroc(
                recall_entropy_at_A[mr], recall_labels[mr])
        else:
            entry["lr_auroc_recall"] = None
            entry["lr_auroc_entropy_only_recall"] = None
        table[name] = entry
    return table
