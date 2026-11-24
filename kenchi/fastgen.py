"""Fast batched generation: one parallel pass over the prompts, then cached steps.

Why this exists (lead, 2026-11-19): the model's own step path (`Kenchi.generate`,
`_step_token`) feeds the prompt one token per step and grows the attention cache
with `torch.cat` at every step.  With real search results in the prompt (1,000+
tokens, 128 rows) that cost about half a second per step on an H100 and filled
the GPU with dead cache copies; the second post-training job ran out of time in
its first measurement.

Here the prompts (any lengths, right-padded) go through the layers ONCE in
parallel, which also yields every layer's state at each row's own last token:

* attention: the rotated keys and the values of all positions, stored in a
  buffer that is allocated once with room for the new tokens;
* Mamba-2: the state after the last real token (padded positions get ``dt = 0``,
  which leaves the state untouched) and the last ``d_conv - 1`` inputs of the
  causal convolution.

After that only the new tokens are stepped, each row at its own position.
The result equals the step path token for token (tests/test_fastgen.py).

Only the released layout is supported: layers of mixer "A" or "M", no loops, no
JEPA branch.  `supported(model)` says whether a model qualifies.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .mixers import A, M, _autocast_disabled, _rotate_half, ssd_chunk

__all__ = ["supported", "prefill", "step", "generate_ids"]


def supported(model) -> bool:
    cfg = model.cfg
    if cfg.loop_end > cfg.loop_start or cfg.jepa != "none":
        return False
    return all(isinstance(layer.mixer, (A, M)) for layer in model.layers)


# ------------------------------------------------------------------ attention

def _rope_at(mixer: A, x, pos):
    """RoPE for one token per row at per-row positions ``pos`` (B,); x is (B, H, 1, d_h)."""
    inv = 1.0 / (10000.0 ** (torch.arange(0, mixer.d_h, 2, device=x.device,
                                           dtype=torch.float32) / mixer.d_h))
    emb = pos.to(torch.float32)[:, None] * inv[None, :]
    emb = torch.cat((emb, emb), dim=-1)                      # (B, d_h)
    cos = emb.cos().to(x.dtype)[:, None, None, :]
    sin = emb.sin().to(x.dtype)[:, None, None, :]
    return x * cos + _rotate_half(x) * sin


def _a_prefill(mixer: A, h, cap):
    B, T, _ = h.shape
    q, k, v = mixer._proj(h)
    q = mixer._rope(q, T)
    k = mixer._rope(k, T)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    y = mixer.out_proj(o.transpose(1, 2).reshape(B, T, -1))
    kc = torch.zeros(B, mixer.n_head, cap, mixer.d_h, dtype=k.dtype, device=k.device)
    vc = torch.zeros(B, mixer.n_head, cap, mixer.d_h, dtype=v.dtype, device=v.device)
    kc[:, :, :T] = k
    vc[:, :, :T] = v
    return y, [kc, vc]


def _a_step(mixer: A, h_t, state, pos, rows):
    """h_t (B, d); writes this token's key/value at ``pos`` and attends to 0..pos."""
    kc, vc = state
    B = h_t.shape[0]
    q, k, v = mixer._proj(h_t.unsqueeze(1))
    q = _rope_at(mixer, q, pos)
    k = _rope_at(mixer, k, pos)
    kc[rows, :, pos] = k[:, :, 0].to(kc.dtype)
    vc[rows, :, pos] = v[:, :, 0].to(vc.dtype)
    used = int(pos.max()) + 1
    mask = (torch.arange(used, device=pos.device)[None, :] <= pos[:, None])[:, None, None, :]
    o = F.scaled_dot_product_attention(q.to(kc.dtype), kc[:, :, :used], vc[:, :, :used],
                                       attn_mask=mask)
    return mixer.out_proj(o.transpose(1, 2).reshape(B, -1).to(h_t.dtype))


# ------------------------------------------------------------------ Mamba-2

def _m_prefill(mixer: M, h, valid, lengths, rows):
    B, T, _ = h.shape
    z, xv, Bv, Cv, dtv = mixer._prep_split(h)
    pre = torch.cat([xv, Bv, Cv], dim=-1)                    # (B, T, C)
    u = F.silu(mixer.conv(pre.transpose(1, 2))[..., :T].transpose(1, 2))
    xv_s, Bv_s, Cv_s = u.split([mixer.d_inner, mixer.d_state, mixer.d_state], dim=-1)
    v = xv_s.view(B, T, mixer.nheads, mixer.head_dim)
    k = Bv_s.unsqueeze(2).expand(B, T, mixer.nheads, mixer.d_state)
    q = Cv_s.unsqueeze(2).expand(B, T, mixer.nheads, mixer.d_state)
    # dt = 0 on padded positions: decay exp(0) = 1 and no input, so the state
    # after the scan is each row's state at its own last real token
    dt = F.softplus(dtv + mixer.b_dt) * valid[:, :, None].to(dtv.dtype)
    log_a = -torch.exp(mixer.A_log) * dt
    with _autocast_disabled(v.device):
        o, S = ssd_chunk(v.float(), k.float(), q.float(), dt.float(), log_a.float(),
                         mixer.chunk)
    y = mixer._tail(o, v, z)
    # the convolution's memory: the last d_conv - 1 inputs before each row's end
    w = mixer.d_conv - 1
    idx = lengths[:, None] - w + torch.arange(w, device=h.device)[None, :]     # (B, w)
    buf = pre[rows[:, None], idx.clamp(min=0)] * (idx >= 0)[:, :, None].to(pre.dtype)
    ref = mixer.out_proj.weight
    return y, (S.to(ref.dtype), buf.transpose(1, 2).to(ref.dtype))


# ------------------------------------------------------------------ model level

@torch.no_grad()
def prefill(model, tokens, lengths, room, return_features=False):
    """Run right-padded ``tokens`` (B, T) with real ``lengths`` (B,) through the model.

    Returns ``(logits, states)``: the next-token logits at each row's last real
    token and the per-layer states for `step`; ``room`` is how many further
    tokens the attention buffers must hold.
    """
    B, T = tokens.shape
    rows = torch.arange(B, device=tokens.device)
    valid = torch.arange(T, device=tokens.device)[None, :] < lengths[:, None]
    cap = T + int(room)
    x = model.tok_emb(tokens)
    states = []
    for layer in model.layers:
        h = layer.norm1(x)
        if isinstance(layer.mixer, A):
            y, st = _a_prefill(layer.mixer, h, cap)
        else:
            y, st = _m_prefill(layer.mixer, h, valid, lengths, rows)
        x = x + y
        x = x + layer.mlp(layer.norm2(x))
        states.append(st)
    last = model.final_norm(x[rows, lengths - 1])
    if return_features:
        return model.lm_head(last), states, last
    return model.lm_head(last), states


@torch.no_grad()
def step(model, token, states, pos, return_features=False):
    """Feed one token per row (B,) at per-row positions ``pos`` (B,); returns the
    next-token logits.  The caller advances ``pos`` afterwards."""
    rows = torch.arange(token.shape[0], device=token.device)
    x = model.tok_emb(token)
    for i, layer in enumerate(model.layers):
        h = layer.norm1(x)
        if isinstance(layer.mixer, A):
            y = _a_step(layer.mixer, h, states[i], pos, rows)
        else:
            y, states[i] = layer.mixer.step(h, states[i])
        x = x + y
        x = x + layer.mlp(layer.norm2(x))
    x = model.final_norm(x)
    if return_features:
        return model.lm_head(x), x
    return model.lm_head(x)


@torch.no_grad()
def generate_ids(model, prompts, *, max_new_tokens, end_id, temperature=0.0, top_k=0,
                 pad_id=None, return_features=False, ban_first=None, prefer_first=None):
    """Generate for a list of token-id lists of any lengths in one batch.

    Returns ``(ids, logprobs)``: per prompt the generated ids (through the first
    ``end_id`` inclusive, or ``max_new_tokens`` of them) and their log-probabilities.
    Call it inside the caller's autocast context.
    """
    dev = next(model.parameters()).device
    B = len(prompts)
    lens = [len(p) for p in prompts]
    if min(lens) < 1:
        raise ValueError("empty prompt")
    T = max(lens)
    M_ = int(max_new_tokens)
    pad = int(end_id if pad_id is None else pad_id)
    P = torch.full((B, T), pad, dtype=torch.long)
    for i, p in enumerate(prompts):
        P[i, :len(p)] = torch.as_tensor(list(p), dtype=torch.long)
    P = P.to(dev)
    L = torch.tensor(lens, dtype=torch.long, device=dev)
    logits, states, feat = prefill(model, P, L, M_, return_features=True)
    feat_pre = feat.float().clone()            # final-layer features at the last prompt token
    feat_last = feat.float().clone()           # ... those the last written token was predicted from
    feat_sum = torch.zeros_like(feat_pre)      # ... and the sum over the positions of the written tokens
    pos = L.clone()
    gen = torch.full((B, M_), pad, dtype=torch.long, device=dev)
    glp = torch.zeros((B, M_), dtype=torch.float32, device=dev)
    count = torch.zeros(B, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    if ban_first:                                  # token ids that may not open the reply
        logits = logits.clone()
        logits[:, list(ban_first)] = float("-inf")
    if prefer_first:
        # prefer_first = (ids_a, ids_b, ratio): if the reply is more likely to open with ANY token of
        # group b than with any of group a (times ratio), group a may not open it.  Used for the
        # evidence line: "No result states this." (a) against "[1] says", "[2] says", ... (b), whose
        # probability is split over the result numbers and would lose to a single "No" otherwise.
        ids_a, ids_b, ratio = prefer_first
        pr = F.softmax(logits.float(), dim=-1)
        if float(pr[:, list(ids_b)].sum()) > float(ratio) * float(pr[:, list(ids_a)].sum()):
            logits = logits.clone()
            logits[:, list(ids_a)] = float("-inf")
    for t in range(M_):
        tok = model._sample(logits, temperature, top_k)
        lp = F.log_softmax(logits.float(), dim=-1).gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        live = ~done
        gen[:, t] = torch.where(live, tok, gen[:, t])
        glp[:, t] = torch.where(live, lp, glp[:, t])
        count = count + live.to(count.dtype)
        done = done | (tok == int(end_id))
        last_step = t == M_ - 1 or bool(done.all())
        if last_step and not return_features:
            break
        # feed the token: its features are at the position of a written token (also for the token
        # that ended the row), so they go into the sum for every row that was still writing
        logits, feat = step(model, tok, states, pos, return_features=True)
        feat_sum = feat_sum + feat.float() * live[:, None]
        if last_step:                              # the extra step for the sum only
            break
        feat_last = torch.where(done[:, None], feat_last, feat.float())
        pos = pos + 1
    gen_c, glp_c, count_c = gen.cpu().tolist(), glp.cpu().tolist(), count.cpu().tolist()
    out = ([gen_c[i][:count_c[i]] for i in range(B)], [glp_c[i][:count_c[i]] for i in range(B)])
    if return_features == "all":
        return out + (feat_pre, feat_last, feat_sum / count.clamp(min=1)[:, None].float())
    if return_features:
        return out + (feat_pre, feat_last)
    return out
