"""FLOP accounting (src/kenchi/flops.py, docs/PHASE1_DESIGN.md §2).

- ``forward_macs_per_token(cfg, T)`` -- analytic multiply-accumulates per token, summed
  over the *unrolled* layers (a looped layer counts once per application; with loop
  sampling use the expected count), including the unembedding (d*V) and the aux heads;
  attention counts the full ``2*T*d`` quadratic term. L/M/D are counted as their linear
  weights plus the chunked kernel actually used in training (intra-chunk quadratic of
  size ``ssd_chunk`` and the inter-chunk recurrent state).
- ``train_flops_per_token(cfg, T)`` = ``2 * macs * 3`` (forward + backward ≈ 3×
  forward), plus ``2 * trunk_macs`` for the EMA target forward when ``jepa == "ema"``.
- ``count_params(model)`` -> unique parameters, split into embedding / body / heads.
"""

from __future__ import annotations

from kenchi.mixers import round_to_64

__all__ = ["forward_macs_per_token", "train_flops_per_token", "count_params",
           "trunk_macs_per_token"]


def _swiglu_hidden(cfg):
    return round_to_64(cfg.mlp_ratio * cfg.d_model)


def _ssd_kernel_macs(n_heads, dv, dk, BT):
    """MACs per token of the chunked SSD kernel (mixers.ssd_chunk), per the einsums it
    runs: S·q (dv*dk), intra-chunk scores q·k (BT*dk), scores·v (BT*dv), and the chunk
    state update v k^T (dv*dk); all per head."""
    return n_heads * (2 * dv * dk + BT * (dk + dv))


def _delta_kernel_macs(n_heads, dv, dk, BT):
    """MACs per token of the chunked gated-delta kernel (mixers._d_kernel), per head:
    Gram matrices k·k and q·k (2*BT*dk), the unit-triangular solve with dv+dk right-hand
    sides (≈ BT*(dv+dk)/2 on average per token), M·R0 (BT*dv) and M·Kt (BT*dk), the
    per-chunk transition Kt^T·dK (dk*dk) and offset R0^T·dK (dv*dk), the state scan
    S·A (dv*dk*dk/BT), and the inter-chunk output Qt·S^T (dk*dv)."""
    return n_heads * (BT * (3 * dk + dv) + BT * (dv + dk) / 2 + dk * dk + 2 * dv * dk
                      + dv * dk * dk / BT)


def _mixer_macs(cfg, ch, T, BT):
    """Analytic MACs per token for one application of mixer `ch` on the TRAINING path
    (the chunked kernels for L / D / M, full T x T attention for A). Elementwise work
    (norms, RoPE, gates, exp/cumsum) is not counted, as in the usual convention."""
    d = cfg.d_model
    H = cfg.n_head
    d_h = d // H
    if ch == "A":
        # q,k,v,o projections + QK^T and AV over the full T x T matrix (2*T*d per token)
        return 4 * d * d + 2 * T * d
    if ch == "L":
        # in_proj (3d^2) + out_proj (d^2) + dt projection (d*H) + SSD kernel
        return 4 * d * d + d * H + _ssd_kernel_macs(H, d_h, d_h, BT)
    if ch == "D":
        # in_proj (3d^2) + out_proj (d^2) + beta and dt projections (2*d*H) + delta kernel
        return 4 * d * d + 2 * d * H + _delta_kernel_macs(H, d_h, d_h, BT)
    if ch == "M":
        d_inner = round_to_64(4 * d / 3)
        ds = int(cfg.mamba_d_state)
        nh = d_inner // 64
        m = d * (2 * d_inner + 2 * ds + nh)               # in_proj: (z, x, B, C, dt)
        m += (d_inner + 2 * ds) * int(cfg.mamba_d_conv)    # causal depthwise conv over [x, B, C]
        m += d_inner * d                                  # out_proj: d_inner -> d
        m += _ssd_kernel_macs(nh, 64, ds, BT)             # SSD kernel
        return m
    raise ValueError(f"unknown mixer {ch!r}")


def _body_macs(cfg, T, upto=None):
    """MACs per token of the trunk: unrolled mixer + SwiGLU per layer, loop_emb adds.
    ``upto`` restricts the count to unique layers [0, upto)."""
    d = cfg.d_model
    swiglu = 3 * d * _swiglu_hidden(cfg)
    BT = int(cfg.ssd_chunk)
    R = cfg.loop_end - cfg.loop_start
    if R > 0:
        if cfg.loop_sample_min > 0:
            n_eff = (cfg.loop_sample_min + cfg.n_loops) / 2
        else:
            n_eff = cfg.n_loops
    else:
        n_eff = 1
    macs = 0.0
    for i, ch in enumerate(cfg.layout if upto is None else cfg.layout[:upto]):
        a = float(n_eff) if (R > 0 and cfg.loop_start <= i < cfg.loop_end) else 1.0
        macs += a * (_mixer_macs(cfg, ch, T, BT) + swiglu)
        if R > 0 and cfg.loop_start <= i < cfg.loop_end:
            macs += a * d                        # loop_emb vectors (§1.3)
    return macs


def forward_macs_per_token(cfg, T=None) -> int:
    """Analytic forward MACs per token (spec §2)."""
    d = cfg.d_model
    V = cfg.vocab_size
    T = int(cfg.seq_len if T is None else T)
    macs = _body_macs(cfg, T)
    macs += d * V                                # unembedding (tied)
    if cfg.u_head:
        h = d // 2
        macs += d * h + h * 2                    # u_head: Linear(d, d/2), Linear(d/2, 2)
    if cfg.jepa != "none":
        macs += d * d + d * 2 * d                # jepa_predictor: Linear(d, d), Linear(d, 2d)
        if cfg.jepa_layer > 0:
            macs += 2 * d * d                    # jepa_inject: Linear(2d, d)
    if cfg.mtp:
        frac = 1.0 / max(1, int(round(1.0 / float(cfg.mtp_frac))))
        macs += frac * (d * d + d * V)           # mtp_mlp + extra unembedding on a 1/stride subset (§1.6)
    return int(macs)


def trunk_macs_per_token(cfg, T=None) -> int:
    """Analytic trunk MACs per token (mixer + SwiGLU + loop_emb), no heads/unembedding."""
    return int(_body_macs(cfg, int(cfg.seq_len if T is None else T)))


def train_flops_per_token(cfg, T=None) -> int:
    """``2 * macs * 3`` (forward + backward ≈ 3× forward) + ``2 * macs_trunk`` for the
    EMA target forward when ``jepa == "ema"`` (spec §2)."""
    flops = 2 * 3 * forward_macs_per_token(cfg, T)
    if cfg.jepa == "ema":
        # the target encoder runs the whole trunk (auxiliary mode) or only the context
        # encoder, layers [0, jepa_layer) (structural mode)
        upto = cfg.jepa_layer if cfg.jepa_layer > 0 else None
        flops += 2 * int(_body_macs(cfg, int(cfg.seq_len if T is None else T), upto))
    return flops


def count_params(model) -> dict:
    """Unique parameters of `model`, split into embedding / body / heads.

    embedding: ``tok_emb.weight`` (the tied ``lm_head.weight`` is the same tensor, so
    it is not double-counted); heads: u_head / jepa_predictor / jepa_ln / mtp_mlp /
    lm_head; body: the blocks, final norm and loop_emb.
    """
    emb = body = heads = 0
    for name, p in model.named_parameters(remove_duplicate=True):
        n = p.numel()
        if name.startswith("tok_emb"):
            emb += n
        elif name.startswith(("u_head", "jepa_predictor", "jepa_inject", "mtp_mlp", "lm_head")):
            heads += n
        else:
            body += n
    return {"embedding": emb, "body": body, "heads": heads,
            "total": emb + body + heads}
