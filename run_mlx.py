#!/usr/bin/env python3
"""Kenchi chat runner for Apple Silicon (MLX). Needs: mlx, numpy, sentencepiece.

    python run_mlx.py --weights model.safetensors --config config.json \
        --question "Who wrote Frankenstein?"

Same behaviour and options as run.py (see there): the chat, `--history last|compact|all`, the
search loop, the evidence line (`--show-thinking`) and the self-check (`--no-self-check` turns it off).

The model is written here as plain functions over the weight table, following kenchi/model.py and
kenchi/mixers.py: 22 blocks, attention ("A") and Mamba-2 ("M") in turn, each followed by a SwiGLU
MLP.  The prompt goes through the blocks once; after that each new token is one small step that
reuses what the blocks remember (attention: the keys and values of all earlier tokens; Mamba-2: its
state and the last inputs of its convolution).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kenchi import format_v2  # noqa: E402
from kenchi.assistant import HISTORY_MODES, Assistant, default_history, read_metadata  # noqa: E402
from search_tools import load_search  # noqa: E402


# ------------------------------------------------------------------ model

def load_weights(path):
    """The safetensors file as a table of MLX arrays (float32), in the layout MLX wants."""
    W = {}
    for name, w in mx.load(str(path)).items():
        if name == "lm_head.weight":
            continue                                          # the same table as tok_emb.weight
        w = w.astype(mx.float32)
        if w.ndim == 2 and name != "tok_emb.weight":
            w = w.T                                           # PyTorch Linear (out, in) -> (in, out)
        W[name] = w
    W["out.weight"] = W["tok_emb.weight"].T                   # the output layer is the embedding table
    mx.eval(W)
    return W


def linear(x, w):
    return x @ w


def silu(x):
    return x * mx.sigmoid(x)


def rmsnorm(x, gain, eps=1e-6):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps) * gain


def rope(x, start=0):
    """Rotary position embedding on (B, H, T, d_h) for positions start..start+T-1 (kenchi/mixers.py A._rope)."""
    t_len, d_h = x.shape[2], x.shape[3]
    inv = 1.0 / (10000.0 ** (mx.arange(0, d_h, 2).astype(mx.float32) / d_h))
    ang = (mx.arange(t_len).astype(mx.float32) + float(start))[:, None] * inv[None, :]
    ang = mx.concatenate([ang, ang], axis=-1)                 # (T, d_h)
    x1, x2 = x[..., : d_h // 2], x[..., d_h // 2:]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return x * mx.cos(ang) + rotated * mx.sin(ang)


def attention(W, p, x, n_head, state=None, pos=0):
    """Causal softmax attention (mixer A).

    Without `state`: the whole sequence x (B, T, d); returns (output, [keys, values]).
    With `state`: x is the one new token (B, 1, d) at position `pos`; its key and value are added."""
    b, t, d = x.shape
    d_h = d // n_head

    def heads(name):
        return linear(x, W[p + name + ".weight"]).reshape(b, t, n_head, d_h).transpose(0, 2, 1, 3)

    q, k, v = rope(heads("q_proj"), pos), rope(heads("k_proj"), pos), heads("v_proj")
    if state is not None:
        k, v = mx.concatenate([state[0], k], axis=2), mx.concatenate([state[1], v], axis=2)
    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(d_h)
    if state is None:
        scores = scores + mx.triu(mx.full((t, t), float("-inf")), k=1)   # no looking ahead
    out = mx.softmax(scores, axis=-1) @ v
    return linear(out.transpose(0, 2, 1, 3).reshape(b, t, d), W[p + "out_proj.weight"]), [k, v]


def ssd(v, k, q, dt, log_a, chunk=64):
    """The Mamba-2 state recurrence for a whole sequence (kenchi/mixers.py ssd_chunk):

        S_t = exp(log_a_t) * S_{t-1} + dt_t * v_t k_t^T        o_t = S_t q_t

    computed chunk by chunk: inside a chunk with matrices, between chunks with
    a short loop. v is (B, T, H, dv); k and q are (B, T, H, dk); dt and log_a (B, T, H).
    Returns the outputs and the state after the last step (B, H, dv, dk).
    """
    b, t, h, dv = v.shape
    pad = (-t) % chunk
    n = (t + pad) // chunk
    if pad:
        v, k, q, dt, log_a = (mx.concatenate([a, mx.zeros((b, pad) + tuple(a.shape[2:]))], axis=1)
                              for a in (v, k, q, dt, log_a))

    def chunks(a):                                            # (B, n*C, H, ...) -> (B, H, n, C, ...)
        a = a.reshape(b, n, chunk, h, *a.shape[3:])
        return a.transpose(0, 3, 1, 2, *range(4, a.ndim))

    vd, kc, qc = chunks(v * dt[..., None]), chunks(k), chunks(q)
    p = mx.cumsum(chunks(log_a), axis=-1)                     # (B, H, n, C) decay from the chunk start
    diff = p[..., :, None] - p[..., None, :]                  # decay from step j to step t
    lower = mx.tril(mx.ones((chunk, chunk))) > 0              # j <= t
    decay = mx.where(lower, mx.exp(mx.where(lower, diff, 0.0)), 0.0)
    out = (decay * (qc @ kc.swapaxes(-1, -2))) @ vd           # contributions from inside the chunk
    to_end = mx.exp(p[..., -1:] - p)                          # decay from step j to the chunk end
    added = (vd * to_end[..., None]).swapaxes(-1, -2) @ kc    # what each chunk adds to the state
    whole = mx.exp(p[..., -1])                                # decay over a whole chunk
    state = mx.zeros((b, h, dv, kc.shape[-1]))
    starts = []
    for c in range(n):
        starts.append(state)
        state = whole[:, :, c, None, None] * state + added[:, :, c]
    start = mx.stack(starts, axis=2)                          # state entering each chunk
    out = out + mx.exp(p)[..., None] * (qc @ start.swapaxes(-1, -2))
    return out.transpose(0, 2, 3, 1, 4).reshape(b, n * chunk, h, dv)[:, :t], state


def mamba(W, p, x, state=None):
    """Mamba-2 (mixer M). Sizes are read from the weights.

    Without `state`: the whole sequence; returns (output, [state, the last inputs of the convolution]).
    With `state`: x is the one new token (B, 1, d); one step of the same recurrence."""
    b, t, _ = x.shape
    d_inner = W[p + "norm.gain"].shape[0]
    n_head = W[p + "A_log"].shape[0]
    width = W[p + "conv.bias"].shape[0]                       # d_inner + 2 * d_state
    d_state = (width - d_inner) // 2
    conv_w = W[p + "conv.weight"]                             # (width, 1, k): one small filter per channel
    k_conv = conv_w.shape[2]

    y = linear(x, W[p + "in_proj.weight"])
    z, xv, bv, cv, dtv = mx.split(y, [d_inner, 2 * d_inner, 2 * d_inner + d_state,
                                      2 * d_inner + 2 * d_state], axis=-1)
    # causal depthwise convolution over time: each channel mixes its last k steps
    u = mx.concatenate([xv, bv, cv], axis=-1)
    before = state[1] if state is not None else mx.zeros((b, k_conv - 1, width))
    u = mx.concatenate([before, u], axis=1)
    memory = u[:, u.shape[1] - (k_conv - 1):]                 # what the next step needs
    u = sum(u[:, j:j + t] * conv_w[:, 0, j] for j in range(k_conv)) + W[p + "conv.bias"]
    u = silu(u)
    xv, bv, cv = mx.split(u, [d_inner, d_inner + d_state], axis=-1)

    v = xv.reshape(b, t, n_head, d_inner // n_head)
    dt = mx.logaddexp(dtv + W[p + "b_dt"], 0.0)               # softplus
    log_a = -mx.exp(W[p + "A_log"]) * dt
    if state is None:
        k = mx.broadcast_to(bv[:, :, None, :], (b, t, n_head, d_state))
        q = mx.broadcast_to(cv[:, :, None, :], (b, t, n_head, d_state))
        o, s = ssd(v, k, q, dt, log_a)
    else:                                                     # S = a * S + dt * v k^T ;  o = S q
        v1, dt1 = v[:, 0], dt[:, 0]                           # (B, H, dv), (B, H)
        s = mx.exp(log_a[:, 0])[:, :, None, None] * state[0] \
            + (v1 * dt1[..., None])[..., None] * bv[:, 0][:, None, None, :]
        o = (s @ cv[:, 0][:, None, :, None])[..., 0][:, None]  # (B, 1, H, dv)
    y = (o + W[p + "D_h"][None, None, :, None] * v).reshape(b, t, d_inner)
    return linear(rmsnorm(y * silu(z), W[p + "norm.gain"]), W[p + "out_proj.weight"]), [s, memory]


def blocks(W, cfg, ids, states=None, pos=0):
    """Token ids (B, T) through all blocks -> (last-layer features after the final norm (B, T, d), states).

    With `states` (from an earlier call) `ids` are the new tokens, one at a time (T = 1), at position `pos`."""
    x = W["tok_emb.weight"][ids]
    new_states = []
    for i, kind in enumerate(cfg["layout"]):
        p = f"layers.{i}."
        h = rmsnorm(x, W[p + "norm1.gain"])
        st = states[i] if states is not None else None
        y, st = attention(W, p + "mixer.", h, cfg["n_head"], st, pos) if kind == "A" else mamba(W, p + "mixer.", h, st)
        new_states.append(st)
        x = x + y
        h = rmsnorm(x, W[p + "norm2.gain"])
        x = x + linear(silu(linear(h, W[p + "mlp.gate.weight"])) * linear(h, W[p + "mlp.up.weight"]),
                       W[p + "mlp.down.weight"])
    return rmsnorm(x, W["final_norm.gain"]), new_states


def forward(W, cfg, ids, last_only=False):
    """Token ids (B, T) -> next-token logits (B, T, vocab), or (B, vocab) for the last position."""
    x, _ = blocks(W, cfg, ids)
    if last_only:
        x = x[:, -1]
    return linear(x, W["out.weight"])


def generate(W, cfg, ids, max_new_tokens=256, temperature=0.0, cached=True, ban_first=None, prefer_first=None):
    """New token ids up to and with `<|end|>`, the last-layer features the last one was predicted from,
    and the mean and the lowest log-probability of the new tokens.

    `cached=False` runs the whole sequence again for every token (slow; the reference for tests)."""
    ids = [int(t) for t in ids]
    out, states, lps = [], None, []
    feats, states = blocks(W, cfg, mx.array([ids]))
    feat = feats[:, -1]
    for step_no in range(int(max_new_tokens)):
        logits = linear(feat, W["out.weight"])[0]
        if step_no == 0 and ban_first:                        # token ids that may not open the reply
            ban = mx.zeros(logits.shape)
            for t in ban_first:
                ban[int(t)] = float("-inf")
            logits = logits + ban
        if step_no == 0 and prefer_first:
            # (ids_a, ids_b, ratio): if any token of group b is more likely to open the reply than any
            # of group a (times ratio), group a may not open it (see kenchi/fastgen.py)
            ids_a, ids_b, ratio = prefer_first
            pr = mx.softmax(logits)
            if sum(float(pr[int(t)].item()) for t in ids_b) > float(ratio) * sum(float(pr[int(t)].item()) for t in ids_a):
                ban = mx.zeros(logits.shape)
                for t in ids_a:
                    ban[int(t)] = float("-inf")
                logits = logits + ban
        if temperature and temperature > 0:
            nxt = int(mx.random.categorical(logits / float(temperature)).item())
        else:
            nxt = int(mx.argmax(logits).item())
        lps.append(float((logits[nxt] - mx.logsumexp(logits)).item()))
        out.append(nxt)
        if nxt == format_v2.END or len(out) >= int(max_new_tokens):
            break
        if cached:
            feats, states = blocks(W, cfg, mx.array([[nxt]]), states, pos=len(ids) + len(out) - 1)
        else:
            feats, _ = blocks(W, cfg, mx.array([ids + out]))
        feat = feats[:, -1]
        mx.eval(feat, states)
    return out, feat, sum(lps) / max(1, len(lps)), (min(lps) if lps else 0.0)


class MlxEngine:
    """The MLX model behind kenchi.assistant.Assistant (the same two methods as its TorchEngine)."""

    def __init__(self, weights, config):
        cfg = json.loads(Path(config).expanduser().read_text())
        self.cfg = cfg.get("model", cfg)
        self.W = load_weights(Path(weights).expanduser())
        self.has_head = "u_head.0.weight" in self.W and "u_head.2.weight" in self.W

    prefer_first_ok = True

    def generate(self, prompt, max_new_tokens, temperature=0.0, ban_first=None, prefer_first=None):
        return generate(self.W, self.cfg, prompt, max_new_tokens, temperature, ban_first=ban_first,
                        prefer_first=prefer_first)

    def check(self, features, channel=0):
        """The self-check head: Linear, GELU, Linear.  Output 0: an answer after a search holds up;
        output 1: a reply without tools holds up."""
        h = linear(features, self.W["u_head.0.weight"])
        h = 0.5 * h * (1.0 + mx.erf(h / math.sqrt(2.0)))
        return float(linear(h, self.W["u_head.2.weight"])[0, channel].item())


def MlxAssistant(weights, config, self_check=True, max_new_tokens=256, history=None, evidence_check=True):
    return Assistant(MlxEngine(weights, config), read_metadata(weights), self_check=self_check, history=history,
                     max_new_tokens=max_new_tokens, evidence_check=evidence_check)


# ------------------------------------------------------------------ chat

def main(argv=None):
    ap = argparse.ArgumentParser(description="Kenchi chat runner (MLX).")
    ap.add_argument("--weights", default="model.safetensors")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--question", default=None, help="ask one question and exit")
    ap.add_argument("--search", default="wikipedia",
                    help="wikipedia (default), none, or file.py:function / module.function")
    ap.add_argument("--system", default=None, help="a system message (default: none; the model knows who it is)")
    ap.add_argument("--no-self-check", action="store_true", help="do not let the self-check head replace answers")
    ap.add_argument("--max-tool-calls", type=int, default=1, help="searches the model may run per question")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--history", nargs="?", const="all", default=None, choices=list(HISTORY_MODES),
                    help="what the model sees of the chat.  compact (the default the released weights ask for in "
                         "config.json): the earlier questions and answers as plain text, without their searches.  "
                         "last: every question is asked alone.  all (also a bare --history): every earlier turn "
                         "with its searches; the model was not trained on that")
    ap.add_argument("--no-evidence-check", action="store_true", help=argparse.SUPPRESS)   # for weights that write an evidence line; the released ones do not
    ap.add_argument("--show-thinking", action="store_true", help=argparse.SUPPRESS)   # for weights that write an evidence line; the released ones do not
    ap.add_argument("--verbose", action="store_true",
                    help="print search results, the self-check score and timings")
    ap.add_argument("--show-results", action="store_true", help=argparse.SUPPRESS)      # the old name of --verbose
    args = ap.parse_args(argv)
    verbose = args.verbose or args.show_results
    show_thinking = args.show_thinking or verbose

    for f in (args.weights, args.config):
        if not Path(f).expanduser().is_file():
            print(f"run_mlx.py: {f} not found. Pass --weights model.safetensors --config config.json "
                  "(both are in the Hugging Face folder of the release).")
            return 1
    bot = MlxAssistant(args.weights, args.config, self_check=not args.no_self_check, max_new_tokens=args.max_new_tokens,
                       history=args.history or default_history(args.config), evidence_check=not args.no_evidence_check)
    backend = load_search(args.search)
    search = (lambda q: backend.search(q)) if backend else None
    system = [{"role": "system", "content": args.system}] if args.system else []
    keep = bot.history != "last"    # "compact" and "all": the chat keeps the conversation
    history = list(system)          # grows only then; in "last" mode every question is asked alone

    def on_event(kind, data):
        if kind == "tool_call":
            print(f"[search] {json.dumps(data['arguments'], ensure_ascii=False)}", flush=True)
        elif kind == "forced_search" and verbose:
            print("[the model answered with citations before searching; searching first]", flush=True)
        elif kind == "tool_result" and verbose:
            print(data, flush=True)
        elif kind == "thinking" and show_thinking:
            print(f"thinking: {data}", flush=True)
        elif kind == "evidence" and verbose:
            print("[evidence] " + ("the line holds" if data["ok"] else "the line does not hold"
                                   + (f"; answer replaced, the model had written: {data['replaced']!r}"
                                      if "replaced" in data else "; answer kept (--no-evidence-check)")), flush=True)
        elif kind == "self_check" and verbose:
            print(f"[self-check] margin {data['margin']:+.1f} (kept when 0 or more): "
                  + ("kept" if data["passed"] else f"replaced; the model had written: {data['replaced']!r}"), flush=True)

    def ask(q):
        t0 = time.time()
        if not keep:
            del history[len(system):]
        history.append({"role": "user", "content": q})
        text, new = bot.ask(history, search=search, max_tool_calls=args.max_tool_calls, on_event=on_event)
        history.extend(new)
        print(text)
        if verbose:
            print(f"[{time.time() - t0:.1f} s]")

    if args.question:
        ask(args.question)
        return 0
    print({"last": "History: last. Each question is asked alone: Kenchi does not remember the one before.",
           "compact": "History: compact. Kenchi sees the earlier questions and answers of this chat, without "
                      "their searches (/new starts over).",
           "all": "History: all. Experimental: Kenchi sees every earlier turn as it was (/new starts over). "
                  "It answers worse this way."}[bot.history])
    while True:
        try:
            q = input("you: ").strip()
        except EOFError:
            return 0
        if q == "/new":
            del history[len(system):]
            continue
        if q:
            ask(q)


if __name__ == "__main__":
    raise SystemExit(main())
