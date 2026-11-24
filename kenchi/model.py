"""The Kenchi model family (src/kenchi/model.py, docs/PHASE1_DESIGN.md §1).

`KenchiConfig` -- the config dataclass with JSON round-trip (`to_dict`/`from_dict`).
`Kenchi` -- the model: ``tok_emb``, one `Block` per character of ``cfg.layout``, final
`RMSNorm`, tied unembedding (`lm_head`), looping (§1.3), the JEPA auxiliary objective
(§1.4, with the EMA target encoder via `update_ema`), the uncertainty head (§1.5),
MTP (§1.6) and the §1.7 forward API.

`StubModel` is an alias of `Kenchi` so the older stub-based tests keep working when the
real model lands.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kenchi.mixers import A, D, L, M, RMSNorm, init_decay_, round_to_64

__all__ = ["Kenchi", "StubModel", "KenchiConfig", "Block", "SwiGLU", "MIXER_TYPES"]

MIXER_TYPES = {"A": A, "L": L, "D": D, "M": M}


# ---------------------------------------------------------------- KenchiConfig


@dataclass
class KenchiConfig:
    """Config for the Kenchi model family (spec §1).

    ``layout`` has one char per unique layer: ``A`` attention, ``L`` scalar-gated
    linear attention, ``D`` gated delta rule, ``M`` Mamba-2 mixer. The default
    ``"AAAAAAAAAAAA"`` is 12 unique layers, all mixer A (the S proxy, spec §3).
    """

    vocab_size: int = 50304
    d_model: int = 384
    n_head: int = 6
    layout: str = "AAAAAAAAAAAA"
    seq_len: int = 1024
    mlp_ratio: float = 8 / 3
    tie_embeddings: bool = True
    # looping: unique layers [loop_start, loop_end) are applied n_loops times (shared)
    loop_start: int = 0
    loop_end: int = 0                # loop_end == loop_start -> no loop
    n_loops: int = 1
    loop_sample_min: int = 0         # >0: train samples n ~ U{min..n_loops} per step
    loop_embed: bool = True          # learned per-iteration vector added at loop entry
    ssd_chunk: int = 64              # chunk length for L / M / D kernels
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    # auxiliary objectives / heads
    jepa: str = "none"               # "none" | "ema" | "sg"
    jepa_span: int = 8
    jepa_weight: float = 0.25
    jepa_ema_decay: float = 0.996
    # 0: JEPA is an auxiliary loss on the last block's output (nothing is fed back).
    # m > 0: structural JEPA ("latent lookahead"): layers [0, m) are the context encoder,
    # the predictor's predicted future latent (mean, log-variance) is injected into the
    # residual stream, and layers [m, L) decode tokens from it. Target = the target
    # encoder's output at the same depth m, averaged over the next jepa_span positions.
    jepa_layer: int = 0
    u_head: bool = True              # uncertainty head on DETACHED features
    u_span: int = 8
    mtp: bool = False                # predict token t+2 as well
    mtp_weight: float = 0.3
    mtp_frac: float = 0.25           # fraction of positions that get the MTP loss

    def __post_init__(self):
        for ch in self.layout:
            if ch not in "ALDM":
                raise ValueError(f"layout char {ch!r} must be one of 'ALDM'")
        if not (0 <= self.loop_start <= self.loop_end <= len(self.layout)):
            raise ValueError("need 0 <= loop_start <= loop_end <= len(layout)")
        if self.n_loops < 1:
            raise ValueError("n_loops must be >= 1")
        if self.ssd_chunk < 1:
            raise ValueError("ssd_chunk must be >= 1")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.vocab_size < 1 or self.d_model < 1:
            raise ValueError("vocab_size and d_model must be >= 1")
        if self.jepa not in ("none", "ema", "sg"):
            raise ValueError("jepa must be 'none', 'ema' or 'sg'")
        if not (0 <= self.jepa_layer < len(self.layout)):
            raise ValueError("need 0 <= jepa_layer < len(layout)")
        if self.jepa_layer and self.loop_start < self.jepa_layer < self.loop_end:
            raise ValueError("jepa_layer must not fall inside the looped segment")

    def to_dict(self):
        """JSON-safe dict of every field (spec §1: config JSON round-trip)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Rebuild a `KenchiConfig` from `to_dict()` output (unknown keys ignored)."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in valid})


# ---------------------------------------------------------------- SwiGLU / Block


class SwiGLU(nn.Module):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` with hidden = round_to_64(mlp_ratio * d)."""

    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm block (spec §1.1): ``x = x + mixer(RMSNorm(x))`` then
    ``x = x + SwiGLU(RMSNorm(x))``."""

    def __init__(self, cfg, mixer):
        super().__init__()
        d = cfg.d_model
        self.mixer = mixer
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d, round_to_64(cfg.mlp_ratio * d))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    def step(self, x_t, state):
        """Cached single-token version of `forward`: returns ``(x_{t+1}, state)``."""
        y, state = self.mixer.step(self.norm1(x_t), state)
        x_t = x_t + y
        return x_t + self.mlp(self.norm2(x_t)), state


def _rel_change(x, prev):
    """Relative residual change: ||x - prev|| / ||prev||, per token (B, T)."""
    num = (x - prev).norm(dim=-1)
    den = prev.norm(dim=-1).clamp_min(1e-12)
    return num / den


def _masked_ce_mean(logits, targets, mask):
    """Mean CE over the positions where `mask` is True (0 when none; branch-free)."""
    # ignore_index keeps shapes static (torch.compile-friendly, no (N, V) gather copy)
    tgt = targets.masked_fill(~mask, -100)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tgt.reshape(-1),
                           ignore_index=-100, reduction="sum")
    return loss / mask.sum().clamp_min(1)


# ---------------------------------------------------------------- Kenchi


class Kenchi(nn.Module):
    """The Kenchi model family (spec §1).

    Forward API (spec §1.7):
    ``out = model(idx, targets=None, loss_mask=None, n_loops=None, return_aux=False,
    return_logits=True)`` with ``out["logits"]`` (B,T,V) unless ``return_logits=False``; with targets: ``out["loss"]`` (total), ``out["loss_lm"]``,
    ``out["loss_jepa"]``, ``out["loss_u"]``, ``out["loss_mtp"]`` when enabled; with
    ``return_aux``: ``out["u_tok_logit"]``, ``out["u_span_logit"]``, ``out["jepa_logvar"]``,
    ``out["jepa_err"]``, ``out["loop_delta"]`` (B,T,n-1).

    ``targets[t]`` is the label scored by ``logits[t]`` (the caller supplies the
    teacher-forced next-token labels; see `ShardLoader` in data.py).
    """

    def __init__(self, cfg: KenchiConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        assert d % cfg.n_head == 0, "d_model must be divisible by n_head"
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.layers = nn.ModuleList([Block(cfg, MIXER_TYPES[ch](cfg)) for ch in cfg.layout])
        self.final_norm = RMSNorm(d)
        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        active = cfg.loop_end > cfg.loop_start
        # learned per-iteration vector added at loop entry (zero init)
        self.loop_emb = nn.Parameter(torch.zeros(cfg.n_loops, d)) if active else None
        if cfg.u_head:
            h = d // 2
            self.u_head = nn.Sequential(nn.Linear(d, h, bias=False), nn.GELU(),
                                        nn.Linear(h, 2, bias=False))
        if cfg.jepa != "none":
            self.jepa_predictor = nn.Sequential(nn.Linear(d, d, bias=False), nn.GELU(),
                                                nn.Linear(d, 2 * d, bias=False))
            if cfg.jepa_layer > 0:
                # feeds the predicted latent (mean, log-variance) back into the stream;
                # zero-init so training starts from the plain model
                self.jepa_inject = nn.Linear(2 * d, d, bias=False)
        if cfg.mtp:
            self.mtp_mlp = nn.Sequential(nn.Linear(d, d, bias=False), nn.GELU())
        self._init_weights()
        # EMA shadow encoder (spec §1.4, "ema"): a full copy of the model, stored in the
        # instance __dict__ so it is NOT registered as a submodule -- it must not appear
        # in parameters() (the optimizer) or the unique-parameter count. It is
        # persisted via state_dict() (see the overrides below).
        if cfg.jepa == "ema":
            ema = copy.deepcopy(self)
            ema.requires_grad_(False)
            object.__setattr__(self, "ema_model", ema)
        else:
            object.__setattr__(self, "ema_model", None)

    def _apply(self, fn, *args, **kwargs):
        # the EMA copy is not a registered submodule, so .to()/.cuda()/.float() must be forwarded
        super()._apply(fn, *args, **kwargs)
        ema = self.__dict__.get("ema_model")
        if ema is not None:
            ema._apply(fn, *args, **kwargs)
        return self

    # ---------------------------------------------------------------- init

    def _init_weights(self):
        # Spec §1.1: normal(0, 0.02) on linear/conv/embedding weights; RMSNorm gains
        # keep the standard gain of 1; loop_emb is zero-init; A_log init so that
        # a_t = exp(-exp(A_log) dt) is in [0.9, 0.999] for typical dt; the residual
        # output projections (mixer out_proj, SwiGLU down) are scaled by
        # 1/sqrt(2 * n_unrolled_layers).
        for m in self.modules():
            if isinstance(m, nn.Linear):          # depthwise convs keep PyTorch's default init
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
        if self.loop_emb is not None:
            nn.init.zeros_(self.loop_emb)
        if hasattr(self, "jepa_inject"):
            nn.init.zeros_(self.jepa_inject.weight)
        for layer in self.layers:
            if hasattr(layer.mixer, "A_log"):
                init_decay_(layer.mixer.A_log)
        n_unrolled = max(1, len(self.cfg.layout) + (self.cfg.loop_end - self.cfg.loop_start)
                         * (self.cfg.n_loops - 1))
        scale = 1.0 / math.sqrt(2 * n_unrolled)
        for layer in self.layers:
            layer.mixer.out_proj.weight.data.mul_(scale)
            layer.mlp.down.weight.data.mul_(scale)

    # ---------------------------------------------------------------- loop count

    def _n_loops(self, n_loops=None):
        cfg = self.cfg
        if cfg.loop_end <= cfg.loop_start:
            return 1
        if n_loops is not None:
            return int(n_loops)
        if self.training and cfg.loop_sample_min > 0:
            lo, hi = int(cfg.loop_sample_min), int(cfg.n_loops)
            if hi > lo:
                return int(torch.randint(lo, hi + 1, ()))
            return lo
        return int(cfg.n_loops)

    # ---------------------------------------------------------------- trunk

    def _run_layers(self, x, lo, hi, n, deltas=None):
        """Apply unique layers [lo, hi) to ``x``; the looped segment (if it lies inside
        the range) is applied ``n`` times. Appends per-iteration residual changes of the
        loop to ``deltas`` when given (§1.3)."""
        cfg = self.cfg
        ls, le = cfg.loop_start, cfg.loop_end
        j = lo
        while j < hi:
            if le > ls and j == ls:
                entry = prev = x
                for i in range(n):
                    if cfg.loop_embed and self.loop_emb is not None:
                        x = x + self.loop_emb[i]
                    for jj in range(ls, le):
                        x = self.layers[jj](x)
                    if deltas is not None and i > 0:
                        deltas.append(_rel_change(x, prev))
                    prev = x
                if deltas is not None and n == 1:
                    deltas.append(_rel_change(x, entry))
                j = le
            else:
                x = self.layers[j](x)
                j += 1
        return x

    def _run_trunk(self, idx, n_loops=None, return_deltas=False, upto=None):
        """Run the unique blocks on token ids (no JEPA injection); returns the output of
        layer ``upto - 1`` (default: the last block, pre final norm) and, when
        `return_deltas`, the per-iteration residual changes of the loop (§1.3)."""
        n = self._n_loops(n_loops)
        deltas = []
        hi = len(self.layers) if upto is None else int(upto)
        x = self._run_layers(self.tok_emb(idx), 0, hi, n, deltas)
        return (x, deltas) if return_deltas else x

    # ---------------------------------------------------------------- forward

    def forward(self, idx, targets=None, loss_mask=None, n_loops=None, return_aux=False,
                return_logits=True):
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B, T = idx.shape
        n_used = self._n_loops(n_loops)
        m = int(self.cfg.jepa_layer) if self.cfg.jepa != "none" else 0
        jepa_out = None
        if m > 0:
            # structural JEPA: encode, predict the future latent, inject it, decode
            loop_deltas = []
            c = self._run_layers(self.tok_emb(idx), 0, m, n_used, loop_deltas)
            jepa_out = self._jepa_terms(idx, c, n_used, upto=m)
            mu, s_ = jepa_out[3], jepa_out[4]
            c = c + self.jepa_inject(torch.cat([mu, s_], dim=-1).to(c.dtype))
            h = self._run_layers(c, m, len(self.layers), n_used, loop_deltas)
        else:
            h, loop_deltas = self._run_trunk(idx, n_used, return_deltas=True)
        x = self.final_norm(h)
        logits = self.lm_head(x)
        out = {"logits": logits}
        mask = torch.ones_like(idx, dtype=torch.bool) if loss_mask is None else loss_mask.to(torch.bool)

        if targets is not None:
            out["loss_lm"] = _masked_ce_mean(logits, targets, mask)
        else:
            out["loss_lm"] = torch.zeros((), device=logits.device, dtype=torch.float32)

        jepa_ok = self.cfg.jepa in ("ema", "sg")
        if jepa_ok:
            if jepa_out is None:
                jepa_out = self._jepa_terms(idx, h, n_used)
            loss_jepa, jepa_logvar, jepa_err = jepa_out[:3]
            out["loss_jepa"] = loss_jepa
        else:
            out["loss_jepa"] = torch.zeros((), device=logits.device, dtype=torch.float32)

        u = None
        if self.cfg.u_head:
            u = self.u_head(x.detach())                     # (B, T, 2)
            if targets is not None:
                out["loss_u"] = self._u_loss(u, targets, logits, mask)
            else:
                out["loss_u"] = torch.zeros((), device=logits.device, dtype=torch.float32)
        else:
            out["loss_u"] = torch.zeros((), device=logits.device, dtype=torch.float32)

        if self.cfg.mtp and targets is not None:
            out["loss_mtp"] = self._mtp_loss(x, targets, mask)
        else:
            out["loss_mtp"] = torch.zeros((), device=logits.device, dtype=torch.float32)

        # total (spec §1.7): loss_lm + jepa_weight*loss_jepa + loss_u + mtp_weight*loss_mtp
        total = out["loss_lm"]
        if jepa_ok:
            total = total + self.cfg.jepa_weight * out["loss_jepa"]
        total = total + out["loss_u"]
        if self.cfg.mtp:
            total = total + self.cfg.mtp_weight * out["loss_mtp"]
        out["loss"] = total

        if not return_logits:
            # training steps do not need the (B, T, V) logits; dropping the reference lets
            # the float32 copies be freed before the backward pass (several GB at V=50k)
            del out["logits"]
        if return_aux:
            if self.cfg.u_head and u is not None:
                out["u_tok_logit"] = u[..., 0]
                out["u_span_logit"] = u[..., 1]
            if jepa_ok:
                out["jepa_logvar"] = jepa_logvar
                out["jepa_err"] = jepa_err
            if self.cfg.loop_end > self.cfg.loop_start and loop_deltas:
                out["loop_delta"] = torch.stack(loop_deltas, dim=2)
        return out

    # ---------------------------------------------------------------- JEPA (§1.4)

    def _jepa_terms(self, idx, h, n_loops=None, upto=None):
        """Returns ``(loss_jepa, jepa_logvar, jepa_err, mu, s)`` per spec §1.4.

        ``h`` is the context: the last-block output (auxiliary mode) or the output of
        layer ``upto - 1`` (structural mode, where ``mu``/``s`` are then injected).

        ``h`` is the online last-block output; for ``jepa == "ema"`` the target is the
        EMA model's own last-block output run with no grad on the same batch (one extra
        forward pass -- counted in flops.py), for ``"sg"`` the online model's own
        last-block output detached. Gradients flow into the trunk through the online
        ``h`` via the predictor's ``mu``/``s``.
        """
        B, T = idx.shape
        k = int(self.cfg.jepa_span)
        n_valid = T - k                      # valid positions t: t <= T - k - 1
        if self.cfg.jepa == "ema":
            with torch.no_grad():
                h_t = self.ema_model._run_trunk(idx, n_loops, upto=upto)
        else:  # "sg"
            h_t = h.detach()
        mu, s = self.jepa_predictor(h).float().chunk(2, dim=-1)      # (B,T,d) each
        s = s.clamp(-6.0, 4.0)
        if n_valid > 0:
            # z_t = layer_norm(mean_{j=1..k} h_t[t+j])
            cum = torch.cumsum(h_t, dim=1)
            t_idx = torch.arange(n_valid, device=idx.device)
            mean_span = (cum[:, t_idx + k] - cum[:, t_idx]) / k
            # parameter-free layer norm: a learnable affine here would let the loss
            # collapse the target to a constant
            z = F.layer_norm(mean_span.float(), (mean_span.shape[-1],))
            mu_v, s_v = mu[:, :n_valid], s[:, :n_valid]
            # 0.5 * mean_d[(z-mu)^2 exp(-s) + s] over valid positions
            nll = 0.5 * ((((z - mu_v) ** 2) * torch.exp(-s_v)) + s_v).mean(dim=-1)
            loss_jepa = nll.mean()
            err = ((z - mu_v) ** 2).mean(-1).detach()
            jepa_err = torch.cat([err, err.new_full((B, T - n_valid), float("nan"))], dim=1)
        else:
            loss_jepa = h.sum() * 0.0
            jepa_err = torch.full((B, T), float("nan"), device=idx.device, dtype=torch.float32)
        jepa_logvar = s.mean(-1)              # mean_d s_t: known at time t
        return loss_jepa, jepa_logvar, jepa_err, mu, s

    # ---------------------------------------------------------------- uncertainty head (§1.5)

    def _u_loss(self, u, targets, logits, mask):
        """BCE of the uncertainty head against the model's own next-token
        correctness indicator: ``u_tok`` at each position, ``u_span`` with the soft
        target = mean over the next ``u_span`` positions of the same indicator."""
        B, T = logits.shape[0], logits.shape[1]
        with torch.no_grad():
            correct = (logits.argmax(-1) == targets).to(torch.float32)    # (B, T)
        mf = mask.to(torch.float32)
        u = u.float()
        loss_tok = (F.binary_cross_entropy_with_logits(u[..., 0], correct, reduction="none")
                    * mf).sum() / mf.sum().clamp_min(1.0)
        k = int(self.cfg.u_span)
        # soft[t] = mean over j=1..k of correct[t+j]; valid where t + k <= T - 1
        cum = torch.cumsum(F.pad(correct, (1, k)), dim=1)                # (B, T+k+1)
        soft = (cum[:, k + 1: k + 1 + T] - cum[:, 1: 1 + T]) / k
        span_ok = mf * ((torch.arange(T, device=logits.device) + k) <= (T - 1))[None, :]
        loss_span = (F.binary_cross_entropy_with_logits(u[..., 1], soft, reduction="none")
                     * span_ok).sum() / span_ok.sum().clamp_min(1.0)
        return loss_tok + loss_span

    # ---------------------------------------------------------------- MTP (§1.6)

    def _mtp_loss(self, x, targets, mask):
        """CE of the MTP logits at position t against the label two steps ahead
        (``targets[t+1]``, i.e. token t+2) on every ``stride``-th position, where
        ``stride = round(1 / mtp_frac)`` (a static subset keeps shapes fixed and makes
        the extra unembedding cost exactly ``mtp_frac * d * V`` per token)."""
        stride = max(1, int(round(1.0 / float(self.cfg.mtp_frac))))
        xs = x[:, :-1:stride]                                # positions t = 0, s, 2s, ... <= T-2
        logits2 = self.lm_head(xs + self.mtp_mlp(xs))        # h2 = h + MLP_small(h); shared unembedding
        tgt2 = targets[:, 1::stride]                         # targets[t+1] = token t+2
        m2 = mask[:, :-1:stride] & mask[:, 1::stride]
        return _masked_ce_mean(logits2, tgt2, m2)

    # ---------------------------------------------------------------- EMA (spec §1.4)

    @torch.no_grad()
    def update_ema(self):
        """EMA update: ``ema = decay * ema + (1 - decay) * online`` over unique params."""
        ema = getattr(self, "ema_model", None)
        if ema is None:
            return
        decay = float(self.cfg.jepa_ema_decay)
        for p, q in zip(self.parameters(), ema.parameters()):
            q.mul_(decay).add_(p.detach().to(q.device), alpha=1.0 - decay)

    # ---------------------------------------------------------------- state_dict

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        ema = getattr(self, "ema_model", None)
        if ema is not None:
            for k, v in ema.state_dict().items():
                sd[f"ema_model.{k}"] = v
        return sd

    def load_state_dict(self, state_dict, strict=True):
        ema_sd = {k[len("ema_model."):]: v for k, v in state_dict.items()
                  if k.startswith("ema_model.")}
        rest = {k: v for k, v in state_dict.items() if not k.startswith("ema_model.")}
        incompatible = super().load_state_dict(rest, strict=strict)
        if ema_sd:
            ema = getattr(self, "ema_model", None)
            if ema is not None:
                ema.load_state_dict(ema_sd, strict=strict)
        return incompatible

    # ---------------------------------------------------------------- cached generation

    def _init_states(self, batch, n_loops=1):
        """Initial recurrent states for `generate`.

        Looped layers [loop_start, loop_end) get one state per loop iteration: each
        application of a looped layer is a fresh pass over the residual sequence (the
        parallel semantics of `_run_trunk`), so the step path keeps per-iteration
        states.
        """
        cfg = self.cfg
        states = []
        for j, layer in enumerate(self.layers):
            if cfg.loop_end > cfg.loop_start and cfg.loop_start <= j < cfg.loop_end:
                states.append([layer.mixer.initial_state(batch) for _ in range(int(n_loops))])
            else:
                states.append(layer.mixer.initial_state(batch))
        return states

    def _step_token(self, token, states, n_loops):
        """One cached step: ``token`` is (B,) ids; returns logits (B, V) for the next
        position, updating ``states`` in place where the mixer supports it."""
        cfg = self.cfg
        ls, le = cfg.loop_start, cfg.loop_end
        m = int(cfg.jepa_layer) if cfg.jepa != "none" else 0
        x = self.tok_emb(token)
        j = 0
        while j < len(self.layers):
            if m > 0 and j == m:
                # structural JEPA: inject the predicted future latent (no target needed)
                mu, s_ = self.jepa_predictor(x).float().chunk(2, dim=-1)
                x = x + self.jepa_inject(torch.cat([mu, s_.clamp(-6.0, 4.0)], dim=-1).to(x.dtype))
            if le > ls and j == ls:
                for i in range(n_loops):
                    if cfg.loop_embed and self.loop_emb is not None:
                        x = x + self.loop_emb[i]
                    for jj in range(ls, le):
                        x, states[jj][i] = self.layers[jj].step(x, states[jj][i])
                j = le
            else:
                x, states[j] = self.layers[j].step(x, states[j])
                j += 1
        return self.lm_head(self.final_norm(x))

    def _sample(self, logits, temperature, top_k):
        if temperature and temperature > 0:
            logits = logits / temperature
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, int(top_k), dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=0):
        """Greedy (``temperature=0``) or temperature/top-k sampled generation using the
        recurrent/cached path (§1.7): the prompt is consumed token by token through
        `Block.step`, then each new token is sampled from the last-step logits."""
        B, T = idx.shape
        n = self._n_loops(None)
        states = self._init_states(B, n)
        appended = []
        logits = None
        for t in range(T):                                   # consume the prompt
            logits = self._step_token(idx[:, t], states, n)
        for _ in range(int(max_new_tokens)):
            tok = self._sample(logits, temperature, top_k)
            appended.append(tok)
            logits = self._step_token(tok, states, n)
        if appended:
            return torch.cat([idx, torch.stack(appended, dim=1)], dim=1)
        return idx


# Backwards-compatible alias: tests/_stub_model.py used this name; `Kenchi` is the
# real model now that W1a has landed.
StubModel = Kenchi
