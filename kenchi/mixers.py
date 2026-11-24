"""Mixers for the Kenchi model family (src/kenchi/mixers.py, docs/PHASE1_DESIGN.md §1.2).

Four mixer types, each ≈4·d² parameters so layouts stay parameter-matched:

- **A** -- causal softmax attention: q,k,v,o projections, RoPE on q,k (base 10000),
  ``F.scaled_dot_product_attention(is_causal=True)``. No QK-norm, no extras.

- **L** -- scalar-gated linear attention ("SSD kernel", RetNet/Mamba-2-style scalar
  gating, not the per-dimension gates of GLA): per head ``S_t = a_t S_{t-1} + dt_t v_t k^T``,
  ``o_t = S_t q_t``, with ``dt_t = softplus(w_dt·x_t + b_dt)`` per head and
  ``a_t = exp(-exp(A_log_h) dt_t) ∈ (0,1)``. ``k`` is L2-normalised per head; no
  positional encoding. Closed form:
  ``o_t = sum_{j<=t} (q_t·k_j) exp(sum_{l=j+1..t} log a_l) dt_j v_j``.
  Provides a parallel chunked path (intra-chunk quadratic of size ``ssd_chunk`` plus the
  inter-chunk recurrent state, "ssd_minimal" of the Mamba-2 paper), an O(T^2) masked
  reference (``_mix_naive``), and a recurrent single-step API (``step``).

- **D** -- gated delta rule (Gated DeltaNet): per head
  ``S_t = a_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T``, ``o_t = S_t q_t``,
  ``k`` L2-normalised, ``beta_t = sigmoid(w_beta·x_t) ∈ (0,1)``, ``a_t`` as in L.
  The chunked path is a pure-torch port of the algorithm of
  ``fla-org/flash-linear-attention`` ``gated_delta_rule`` chunk implementation
  (``fla/ops/gated_delta_rule/chunk.py``, MIT; Songlin Yang et al.): a triangular solve
  for the WY-style residual sequence R and masked intra-chunk matmuls for the outputs.
  The naive recurrence is the ground truth.

- **M** -- Mamba-2 mixer: ``d_inner = round_to_64(4d/3)``, ``nheads = d_inner // 64``;
  ``in_proj: d -> 2·d_inner + 2·d_state + nheads`` giving ``(z, x, B, C, dt)``; causal
  depthwise conv (kernel ``mamba_d_conv``) + SiLU over ``[x, B, C]``; the shared SSD
  kernel with ``B`` as key, ``C`` as query (shared across heads) and ``x`` as value;
  ``y = y + D_h · x``; ``y = RMSNorm(y · SiLU(z))``; ``out_proj: d_inner -> d``.

Every mixer exposes ``forward(x)`` (the training/parallel path; the L/M/D kernels run in
float32 with autocast disabled), ``forward_naive(x)`` (reference), ``step(x_t, state) ->
(y_t, state)`` (cached generation), ``initial_state(batch)``.

The L/M/D "kernels" are dtype-flexible: ``_mix_parallel`` casts inputs to float32 and
wraps in ``torch.autocast(device_type=..., enabled=False)``; tests call the kernels
directly on float64 tensors.
"""

from __future__ import annotations

import contextlib
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ utils


def round_to_64(x) -> int:
    """Round ``x`` to the nearest multiple of 64 (min 64). Spec §1.2/§3."""
    v = int(round(float(x)))
    return max(64, v // 64 * 64)


def _autocast_disabled(device):
    """``torch.autocast(device_type=..., enabled=False)``; falls back to nullcontext on
    device types torch.autocast does not accept (e.g. some MPS builds)."""
    try:
        return torch.autocast(device_type=device.type, enabled=False)
    except (ValueError, TypeError, RuntimeError):
        return contextlib.nullcontext()


class RMSNorm(nn.Module):
    """RMSNorm with learned gain: ``x * rsqrt(mean(x^2) + eps) * gain``."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.gain


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def init_decay_(A_log):
    """Spread the per-head decay rates log-uniformly so that, at dt = softplus(0) ≈ 0.69,
    a = exp(-exp(A_log) * dt) covers roughly [0.90, 0.999] across heads (fast to slow)."""
    H = A_log.numel()
    rates = torch.logspace(math.log10(0.0015), math.log10(0.15), H)
    with torch.no_grad():
        A_log.copy_(rates.log())
    return A_log


# ------------------------------------------------------------------ shared SSD kernel (L, M)


def _to_chunks(x, n, C):
    """(B, n*C, H, ...) -> (B, H, n, C, ...)."""
    B, _, H = x.shape[:3]
    x = x.reshape(B, n, C, H, *x.shape[3:])
    return x.permute(0, 3, 1, 2, *range(4, x.dim()))


def _from_chunks(o, T):
    """(B, H, n, C, d) -> (B, T, H, d) (drops the padding)."""
    B, H, n, C, d = o.shape
    return o.permute(0, 2, 3, 1, 4).reshape(B, n * C, H, d)[:, :T]


def _pad_time(x, pad):
    if pad == 0:
        return x
    return torch.cat([x, x.new_zeros(x.shape[0], pad, *x.shape[2:])], dim=1)


def _chunk_decays(log_a_c):
    """Per-chunk decay terms from chunked ``log_a`` of shape (B, H, n, C).

    Returns ``g`` (B,H,n,C): decay from the chunk start through t, ``exp(P_t - P_start)``;
    ``E`` (B,H,n,C,C): ``exp(P_t - P_j)`` for j <= t and 0 above the diagonal;
    ``d`` (B,H,n,C): decay from j to the chunk end, ``exp(P_end - P_j)``;
    ``gamma`` (B,H,n): whole-chunk decay ``exp(P_end - P_start)``.
    All exponents are <= 0, so nothing can overflow.
    """
    Pl = torch.cumsum(log_a_c, dim=-1)                       # chunk-local inclusive cumsum
    C = Pl.shape[-1]
    tril = torch.ones(C, C, dtype=torch.bool, device=Pl.device).tril()
    diff = Pl[..., :, None] - Pl[..., None, :]               # P_t - P_j
    E = torch.exp(torch.where(tril, diff, torch.full_like(diff, float("-inf"))))
    P_end = Pl[..., -1]
    return torch.exp(Pl), E, torch.exp(P_end[..., None] - Pl), torch.exp(P_end)


def ssd_chunk(v, k, q, dt, log_a, chunk_size):
    """Chunked linear-attention kernel shared by L and M (spec §1.2).

    State per head ``S`` in ``R^{dv x dk}``; update ``S_t = exp(log_a_t) S_{t-1} + dt_t v_t
    k_t^T``; output ``o_t = S_t q_t``; ``log_a_t <= 0``.

    The sequence is cut into chunks of ``chunk_size``. Everything inside a chunk is
    computed for all chunks at once (no Python loop over time):
      intra-chunk   o_t += sum_{j<=t, same chunk} exp(P_t - P_j) (q_t·k_j) dt_j v_j
      chunk update  B_c  = sum_j exp(P_end - P_j) dt_j v_j k_j^T
    and only the chunk-to-chunk state recurrence ``S_{c+1} = gamma_c S_c + B_c`` is a
    short sequential scan (T / chunk_size cheap steps); then
      inter-chunk   o_t += exp(P_t - P_start) S_c q_t.
    Linear in T, exact (matches the O(T^2) reference to float precision).
    Returns ``(o, S_final)``; ``o`` is ``(B, T, H, dv)``.
    """
    B, T, H, dv = v.shape
    C = int(chunk_size)
    pad = (-T) % C
    n = (T + pad) // C
    v, k, q, dt, log_a = (_pad_time(x, pad) for x in (v, k, q, dt, log_a))
    vd = _to_chunks(v * dt[..., None], n, C)                 # (B,H,n,C,dv)  dt_j v_j
    kc, qc = _to_chunks(k, n, C), _to_chunks(q, n, C)        # (B,H,n,C,dk)
    g, E, d, gamma = _chunk_decays(_to_chunks(log_a, n, C))
    # intra-chunk
    G = E * torch.matmul(qc, kc.transpose(-1, -2))           # (B,H,n,C,C)
    o = torch.matmul(G, vd)                                  # (B,H,n,C,dv)
    # chunk-to-chunk state scan
    Bc = torch.matmul((vd * d[..., None]).transpose(-1, -2), kc)   # (B,H,n,dv,dk)
    S = torch.zeros(B, H, dv, kc.shape[-1], dtype=v.dtype, device=v.device)
    starts = []
    for c in range(n):
        starts.append(S)
        S = gamma[:, :, c, None, None] * S + Bc[:, :, c]
    S0 = torch.stack(starts, dim=2)                          # (B,H,n,dv,dk) state entering chunk c
    # inter-chunk
    o = o + g[..., None] * torch.matmul(qc, S0.transpose(-1, -2))
    return _from_chunks(o, T), S


# ------------------------------------------------------------------ A


class A(nn.Module):
    """Causal softmax attention with RoPE (spec §1.2, mixer A)."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        H = cfg.n_head
        assert d % H == 0, "d_model must be divisible by n_head"
        self.n_head = H
        self.d_h = d // H
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def _proj(self, x):
        B, T, d = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_h).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_h).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_h).transpose(1, 2)
        return q, k, v

    def _rope(self, x, T, start=0):
        # GPT-NeoX style RoPE: x*cos + rotate_half(x)*sin, base 10000.
        inv = 1.0 / (10000.0 ** (torch.arange(0, self.d_h, 2, device=x.device,
                                               dtype=torch.float32) / self.d_h))
        t = torch.arange(start, start + T, device=x.device, dtype=torch.float32)
        emb = torch.cat((torch.outer(t, inv),) * 2, dim=-1)   # (T, d_h)
        cos = emb.cos().to(x.dtype)[None, None]
        sin = emb.sin().to(x.dtype)[None, None]
        return x * cos + _rotate_half(x) * sin

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self._proj(x)
        q = self._rope(q, T)
        k = self._rope(k, T)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(o.transpose(1, 2).reshape(B, T, -1))

    def forward_naive(self, x):
        """O(T^2) masked reference (tests)."""
        B, T, _ = x.shape
        q, k, v = self._proj(x)
        q = self._rope(q, T)
        k = self._rope(k, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_h)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()   # j <= i
        scores = scores.masked_fill(~mask, float("-inf"))
        o = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.out_proj(o.transpose(1, 2).reshape(B, T, -1))

    def initial_state(self, batch):
        return (torch.zeros(batch, self.n_head, 0, self.d_h, dtype=self.q_proj.weight.dtype,
                            device=self.q_proj.weight.device),
                torch.zeros(batch, self.n_head, 0, self.d_h, dtype=self.q_proj.weight.dtype,
                            device=self.q_proj.weight.device))

    def step(self, x_t, state):
        """One cached step: ``x_t`` is (B, d); returns ``(y_t, state)`` with ``y_t`` (B, d)."""
        k_cache, v_cache = state
        q, k, v = self._proj(x_t.unsqueeze(1))
        pos = k_cache.shape[2]
        q = self._rope(q, 1, start=pos)
        k = self._rope(k, 1, start=pos)
        k_cache = torch.cat([k_cache, k], dim=2)
        v_cache = torch.cat([v_cache, v], dim=2)
        o = F.scaled_dot_product_attention(q, k_cache, v_cache, is_causal=False)
        B = x_t.shape[0]
        return self.out_proj(o.transpose(1, 2).reshape(B, -1)), (k_cache, v_cache)


# ------------------------------------------------------------------ L


def _l_naive(q, k, v, dt, log_a):
    """O(T^2) masked reference for L: ``S_t = a_t S_{t-1} + dt_t v_t k_t^T``,
    ``o_t = S_t q_t``. Returns ``o`` of shape (B, T, H, dv)."""
    B, T, H, dv = v.shape
    S = torch.zeros(B, H, dv, k.shape[-1], dtype=v.dtype, device=v.device)
    outs = []
    for t in range(T):
        a_t = torch.exp(log_a[:, t])                                  # (B, H)
        dtv = dt[:, t]
        S = a_t[:, :, None, None] * S + dtv[:, :, None, None] * (
            v[:, t][:, :, :, None] * k[:, t][:, :, None, :])
        outs.append(torch.einsum('bhvk,bhk->bhv', S, q[:, t]))
    return torch.stack(outs, dim=1)


class L(nn.Module):
    """Scalar-gated linear attention (SSD kernel), spec §1.2, mixer L."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        H = cfg.n_head
        assert d % H == 0
        self.n_head = H
        self.d_h = d // H
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.w_dt = nn.Parameter(torch.empty(d, H))
        nn.init.normal_(self.w_dt, 0.0, 0.02)
        self.b_dt = nn.Parameter(torch.zeros(H))
        # A_log init so that a_t = exp(-exp(A_log) dt) in [0.9, 0.999] for typical dt.
        self.A_log = nn.Parameter(torch.full((H,), math.log(0.05)))
        self.out_proj = nn.Linear(d, d, bias=False)
        self.norm = RMSNorm(d)
        self.chunk = int(cfg.ssd_chunk)

    def _prep(self, x):
        B, T, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_h)
        k = k.view(B, T, self.n_head, self.d_h)
        v = v.view(B, T, self.n_head, self.d_h)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)          # k L2-normalised per head
        dt = F.softplus(x @ self.w_dt + self.b_dt)                # (B, T, H)
        log_a = -torch.exp(self.A_log) * dt                # (B, T, H) <= 0
        return q, k, v, dt, log_a

    def _mix_naive(self, q, k, v, dt, log_a):
        return _l_naive(q, k, v, dt, log_a)

    def _mix_parallel(self, q, k, v, dt, log_a):
        # Training path: kernel in float32, autocast disabled (spec §1.2).
        with _autocast_disabled(q.device):
            o, _ = ssd_chunk(v.float(), k.float(), q.float(), dt.float(), log_a.float(),
                             self.chunk)
        return o

    def _tail(self, o):
        B, T = o.shape[0], o.shape[1]
        return self.out_proj(self.norm(o.reshape(B, T, -1)))

    def forward(self, x):
        return self._tail(self._mix_parallel(*self._prep(x)))

    def forward_naive(self, x):
        return self._tail(self._mix_naive(*self._prep(x)))

    def initial_state(self, batch):
        return torch.zeros(batch, self.n_head, self.d_h, self.d_h,
                           dtype=self.out_proj.weight.dtype, device=self.out_proj.weight.device)

    def step(self, x_t, state):
        """One cached step: ``x_t`` is (B, d); returns ``(y_t, state)`` with ``y_t`` (B, d)."""
        B = x_t.shape[0]
        q, k, v, dt, log_a = self._prep_step(x_t)
        S = state if state is not None else torch.zeros(
            B, self.n_head, self.d_h, self.d_h, dtype=v.dtype, device=v.device)
        a = torch.exp(log_a)
        S = a[:, :, None, None] * S + dt[:, :, None, None] * (v[:, :, :, None] * k[:, :, None, :])
        o = torch.einsum('bhvk,bhk->bhv', S, q)
        return self.out_proj(self.norm(o.reshape(B, -1))), S

    def _prep_step(self, x_t):
        B = x_t.shape[0]
        q, k, v = self.in_proj(x_t).chunk(3, dim=-1)
        q = q.view(B, self.n_head, self.d_h)
        k = F.normalize(k.view(B, self.n_head, self.d_h), p=2, dim=-1, eps=1e-6)
        v = v.view(B, self.n_head, self.d_h)
        dt = F.softplus(x_t @ self.w_dt + self.b_dt)
        log_a = -torch.exp(self.A_log) * dt
        return q, k, v, dt, log_a


# ------------------------------------------------------------------ D


def _d_naive(q, k, v, beta, log_a):
    """O(T) recurrent reference for D:
    ``S_t = a_t S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T``, ``o_t = S_t q_t``.
    Returns ``o`` of shape (B, T, H, dv)."""
    B, T, H, dv = v.shape
    dk = k.shape[-1]
    S = torch.zeros(B, H, dv, dk, dtype=v.dtype, device=v.device)
    outs = []
    for t in range(T):
        a_t = torch.exp(log_a[:, t])                       # (B, H)
        b_t = beta[:, t]                                   # (B, H)
        kt = k[:, t]                                       # (B, H, dk)
        vt = v[:, t]                                       # (B, H, dv)
        Sk = torch.einsum('bhvk,bhk->bhv', S, kt)          # (B, H, dv)
        S = a_t[:, :, None, None] * S \
            - (a_t * b_t)[:, :, None, None] * Sk[:, :, :, None] * kt[:, :, None, :] \
            + b_t[:, :, None, None] * vt[:, :, :, None] * kt[:, :, None, :]
        outs.append(torch.einsum('bhvk,bhk->bhv', S, q[:, t]))
    return torch.stack(outs, dim=1)


def _d_kernel(q, k, v, beta, log_a, chunk_size):
    """Chunked gated-delta-rule kernel (see the fla port note in the module docstring).

    Within a chunk (P = chunk-local cumsum of log_a, S_in = state entering the chunk) the
    recurrence unrolls to ``S_t = exp(P_t) S_in + sum_{j<=t} exp(P_t - P_j) R_j k_j^T`` with
      R_t = beta_t v_t - beta_t exp(P_t)(S_in k_t) - sum_{i<t} beta_t exp(P_t - P_i)(k_i·k_t) R_i,
    a unit-lower-triangular system ``Tm R = beta V - diag(beta g) K S_in^T``. Solving it once
    with the two right-hand sides ``beta V`` and ``diag(beta g) K`` gives ``R = R0 - Kt S_in^T``,
    so everything that does not depend on ``S_in`` is computed for all chunks at once:
      outputs     o   = M R0 + (diag(g) Q - M Kt) S_in^T,      M[t,j] = exp(P_t - P_j)(q_t·k_j)
      state       S_out = S_in (gamma I - Kt^T diag(d) K) + R0^T diag(d) K
    and only the small state recurrence is a sequential scan over the T / chunk chunks.
    Returns ``o`` of shape (B, T, H, dv).
    """
    B, T, H, dv = v.shape
    dk = k.shape[-1]
    C = int(chunk_size)
    pad = (-T) % C
    n = (T + pad) // C
    q, k, v, beta, log_a = (_pad_time(x, pad) for x in (q, k, v, beta, log_a))
    qc, kc, vc = _to_chunks(q, n, C), _to_chunks(k, n, C), _to_chunks(v, n, C)
    bc = _to_chunks(beta, n, C)                              # (B,H,n,C)
    g, E, d, gamma = _chunk_decays(_to_chunks(log_a, n, C))
    # unit-lower-triangular system, two right-hand sides at once
    Tm = (bc[..., None] * E * torch.matmul(kc, kc.transpose(-1, -2))).tril(-1)
    rhs = torch.cat([bc[..., None] * vc, (bc * g)[..., None] * kc], dim=-1)
    X = torch.linalg.solve_triangular(Tm, rhs, upper=False, unitriangular=True)
    R0, Kt = X[..., :dv], X[..., dv:]                        # (B,H,n,C,dv), (B,H,n,C,dk)
    M = E * torch.matmul(qc, kc.transpose(-1, -2))           # (B,H,n,C,C)
    o = torch.matmul(M, R0)
    Qt = g[..., None] * qc - torch.matmul(M, Kt)             # (B,H,n,C,dk)
    dK = d[..., None] * kc
    eye = torch.eye(dk, dtype=v.dtype, device=v.device)
    Ac = gamma[..., None, None] * eye - torch.matmul(Kt.transpose(-1, -2), dK)   # (B,H,n,dk,dk)
    Bc = torch.matmul(R0.transpose(-1, -2), dK)                                  # (B,H,n,dv,dk)
    S = torch.zeros(B, H, dv, dk, dtype=v.dtype, device=v.device)
    starts = []
    for c in range(n):
        starts.append(S)
        S = torch.matmul(S, Ac[:, :, c]) + Bc[:, :, c]
    S0 = torch.stack(starts, dim=2)                          # (B,H,n,dv,dk)
    o = o + torch.matmul(Qt, S0.transpose(-1, -2))
    return _from_chunks(o, T)


class D(nn.Module):
    """Gated delta rule (Gated DeltaNet), spec §1.2, mixer D."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        H = cfg.n_head
        assert d % H == 0
        self.n_head = H
        self.d_h = d // H
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.w_beta = nn.Linear(d, H, bias=False)
        self.w_dt = nn.Linear(d, H, bias=False)
        self.b_dt = nn.Parameter(torch.zeros(H))
        # A_log init so that a_t = exp(-exp(A_log) dt) in [0.9, 0.999] for typical dt.
        self.A_log = nn.Parameter(torch.full((H,), math.log(0.05)))
        self.out_proj = nn.Linear(d, d, bias=False)
        self.norm = RMSNorm(d)
        self.chunk = int(cfg.ssd_chunk)

    def _prep(self, x):
        B, T, _ = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_h)
        k = k.view(B, T, self.n_head, self.d_h)
        v = v.view(B, T, self.n_head, self.d_h)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
        beta = torch.sigmoid(self.w_beta(x))                 # (B, T, H)
        dt = F.softplus(self.w_dt(x) + self.b_dt)            # (B, T, H)
        log_a = -torch.exp(self.A_log) * dt
        return q, k, v, beta, log_a

    def _mix_naive(self, q, k, v, beta, log_a):
        return _d_naive(q, k, v, beta, log_a)

    def _mix_parallel(self, q, k, v, beta, log_a):
        with _autocast_disabled(q.device):
            return _d_kernel(q.float(), k.float(), v.float(), beta.float(), log_a.float(),
                             self.chunk)

    def _tail(self, o):
        B, T = o.shape[0], o.shape[1]
        return self.out_proj(self.norm(o.reshape(B, T, -1)))

    def forward(self, x):
        return self._tail(self._mix_parallel(*self._prep(x)))

    def forward_naive(self, x):
        return self._tail(self._mix_naive(*self._prep(x)))

    def initial_state(self, batch):
        return torch.zeros(batch, self.n_head, self.d_h, self.d_h,
                           dtype=self.out_proj.weight.dtype, device=self.out_proj.weight.device)

    def step(self, x_t, state):
        """One cached step: ``x_t`` is (B, d); returns ``(y_t, state)`` with ``y_t`` (B, d)."""
        B = x_t.shape[0]
        q, k, v, beta, log_a = self._prep_step(x_t)
        S = state if state is not None else torch.zeros(
            B, self.n_head, self.d_h, self.d_h, dtype=v.dtype, device=v.device)
        a = torch.exp(log_a)
        Sk = torch.einsum('bhvk,bhk->bhv', S, k)
        S = a[:, :, None, None] * S \
            - (a * beta)[:, :, None, None] * Sk[:, :, :, None] * k[:, :, None, :] \
            + beta[:, :, None, None] * v[:, :, :, None] * k[:, :, None, :]
        o = torch.einsum('bhvk,bhk->bhv', S, q)
        return self.out_proj(self.norm(o.reshape(B, -1))), S

    def _prep_step(self, x_t):
        B = x_t.shape[0]
        q, k, v = self.in_proj(x_t).chunk(3, dim=-1)
        q = q.view(B, self.n_head, self.d_h)
        k = F.normalize(k.view(B, self.n_head, self.d_h), p=2, dim=-1, eps=1e-6)
        v = v.view(B, self.n_head, self.d_h)
        beta = torch.sigmoid(self.w_beta(x_t))
        dt = F.softplus(self.w_dt(x_t) + self.b_dt)
        log_a = -torch.exp(self.A_log) * dt
        return q, k, v, beta, log_a


# ------------------------------------------------------------------ M


class M(nn.Module):
    """Mamba-2 mixer, spec §1.2, mixer M."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        d_inner = round_to_64(4 * d / 3)
        nheads = d_inner // 64
        d_state = int(cfg.mamba_d_state)
        d_conv = int(cfg.mamba_d_conv)
        assert nheads >= 1 and d_inner % 64 == 0
        self.d_inner = d_inner
        self.nheads = nheads
        self.d_state = d_state
        self.d_conv = d_conv
        self.head_dim = 64
        self.in_proj = nn.Linear(d, 2 * d_inner + 2 * d_state + nheads, bias=False)
        C = d_inner + 2 * d_state
        self.conv = nn.Conv1d(C, C, kernel_size=d_conv, groups=C,
                              padding=d_conv - 1, bias=True)
        self.A_log = nn.Parameter(torch.full((nheads,), math.log(0.05)))
        self.b_dt = nn.Parameter(torch.zeros(nheads))
        self.D_h = nn.Parameter(torch.ones(nheads))
        self.out_proj = nn.Linear(d_inner, d, bias=False)
        self.norm = RMSNorm(d_inner)                   # RMSNorm(y * SiLU(z)) on the d_inner stream
        self.chunk = int(cfg.ssd_chunk)

    def _prep_split(self, x):
        """Pre-conv pieces: ``z`` (B,T,d_inner), ``xv``, ``Bv``, ``Cv``, ``dtv``."""
        return self.in_proj(x).split(
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads], dim=-1)

    def _prep(self, x):
        B, T, _ = x.shape
        z, xv, Bv, Cv, dtv = self._prep_split(x)
        u = self.conv(torch.cat([xv, Bv, Cv], dim=-1).transpose(1, 2))[..., :T].transpose(1, 2)
        u = F.silu(u)
        xv_s, Bv_s, Cv_s = u.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        v = xv_s.view(B, T, self.nheads, self.head_dim)
        k = Bv_s.unsqueeze(2).expand(B, T, self.nheads, self.d_state)
        q = Cv_s.unsqueeze(2).expand(B, T, self.nheads, self.d_state)
        dt = F.softplus(dtv + self.b_dt)                     # (B, T, H) > 0
        log_a = -torch.exp(self.A_log) * dt                  # (B, T, H) <= 0
        return z, v, k, q, dt, log_a

    def _mix_naive(self, z, v, k, q, dt, log_a):
        return _l_naive(q, k, v, dt, log_a)

    def _mix_parallel(self, z, v, k, q, dt, log_a):
        with _autocast_disabled(v.device):
            o, _ = ssd_chunk(v.float(), k.float(), q.float(), dt.float(), log_a.float(),
                             self.chunk)
        return o

    def _tail(self, o, v, z):
        B, T = o.shape[0], o.shape[1]
        y = (o + self.D_h[None, None, :, None] * v.to(o.dtype)).reshape(B, T, self.d_inner)
        return self.out_proj(self.norm(y.to(z.dtype) * F.silu(z)))

    def forward(self, x):
        z, v, k, q, dt, log_a = self._prep(x)
        return self._tail(self._mix_parallel(z, v, k, q, dt, log_a), v, z)

    def forward_naive(self, x):
        z, v, k, q, dt, log_a = self._prep(x)
        return self._tail(self._mix_naive(z, v, k, q, dt, log_a), v, z)

    def initial_state(self, batch):
        return (torch.zeros(batch, self.nheads, self.head_dim, self.d_state,
                            dtype=self.out_proj.weight.dtype, device=self.out_proj.weight.device),
                torch.zeros(batch, self.d_inner + 2 * self.d_state, self.d_conv - 1,
                            dtype=self.out_proj.weight.dtype,
                            device=self.out_proj.weight.device))

    def step(self, x_t, state):
        """One cached step: ``x_t`` is (B, d); returns ``(y_t, state)`` with ``y_t`` (B, d)."""
        B = x_t.shape[0]
        z, xv, Bv, Cv, dtv = self._prep_split(x_t.unsqueeze(1))
        S, buf = state if state is not None else self.initial_state(B)
        # causal depthwise conv over the new token with the cached buffer
        seq = torch.cat([buf, torch.cat([xv, Bv, Cv], dim=-1).transpose(1, 2)], dim=-1)
        u = F.silu(F.conv1d(seq, self.conv.weight, self.conv.bias,
                            groups=self.conv.groups))[..., 0]
        buf = seq[:, :, 1:]
        xv_s, Bv_s, Cv_s = u.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        v = xv_s.view(B, self.nheads, self.head_dim)
        k = Bv_s.unsqueeze(1).expand(B, self.nheads, self.d_state)   # shared key across heads
        q = Cv_s.unsqueeze(1).expand(B, self.nheads, self.d_state)  # shared query across heads
        dt = F.softplus(dtv.squeeze(1) + self.b_dt)            # (B, H)
        log_a = -torch.exp(self.A_log) * dt                    # (B, H)
        a = torch.exp(log_a)
        S = a[:, :, None, None] * S + dt[:, :, None, None] * (v[:, :, :, None] * k[:, :, None, :])
        o = torch.einsum('bhvk,bhk->bhv', S, q)
        y = (o + self.D_h[None, :, None] * v).reshape(B, self.d_inner)
        return self.out_proj(self.norm(y * F.silu(z[:, 0]))), (S, buf)
