"""Phase 2 pretraining trainer (src/kenchi/pretrain.py).

CLI:
    python -m kenchi.pretrain --config configs/phase2/base.json [--override k=v ...]
    torchrun --standalone --nproc-per-node=8 scripts/launch/pretrain_run.py --config ...

Design (docs-free summary of the contract):

- **Token schedule** (``TokenSchedule``): the schedule is in TOKENS, not steps:
  the config gives ``total_tokens``, ``warmup_tokens``, peak lr, final lr
  fraction and shape ("cosine" | "linear"); after warm-up LR = peak *
  ((1 - final_frac) * decay(progress) + final_frac).
- **Time budget** (``--time_budget_s`` / ``time_budget_s``): when set, the
  trainer measures throughput after a warm-up period and re-plans
  ``total_tokens`` at fixed points of the budget so the schedule ends when the
  time budget ends; replans never move LR discontinuously (progress is
  re-anchored, see ``TokenSchedule.replan``). Every replan is logged as a
  ``REPLAN_JSON`` line. Needed because the node is billed by the hour and is
  preemptible.
- **Data**: ``kenchi.shard_stream.ShardStream`` (phases of sources + weights).
- **Checkpoints** (``kenchi.ckpt.Checkpointer``): periodic "resume" checkpoints
  every ``ckpt_every_s`` (bf16 weights + bf16 AdamW state + step/tokens/schedule/
  stream/RNG/config; one background serialise+upload thread; a checkpoint is
  skipped while the previous upload is running); final artifacts at the end:
  ``<run>_final.safetensors`` (fp32) + ``<run>_final_config.json`` +
  ``resume_<run>_final.pt`` (fp32), uploaded synchronously with retries until
  the relay confirms (sha+size verified by the client); SIGTERM saves a resume
  checkpoint best-effort.
- **Rank coordination** (A): rank 0 is the only decision maker (stop / re-plan /
  divergence / SIGTERM); a 5-float control vector is broadcast each step and
  every rank applies it before computing the LR; the NaN check is an all-reduce
  before backward so DDP ranks stay in lockstep.
- **Compile order** (C): build the model -> ``torch.compile`` -> DDP;
  ``update_ema``, validation, checkpoints and param counting use the unwrapped
  module (``ckpt.unwrap``).
- **Logging**: JSON lines on stdout (rank 0) every ``log_every`` steps
  (``LOG_JSON`` with step/tokens/loss/aux/lr/grad_norm/tok_per_s/elapsed/phase/
  gpu_mem/ETA), ``VAL_JSON`` on the validation shard every ``val_every_tokens``,
  ``START_JSON`` and ``RESULT_JSON`` (+ the ``RESULT_B64`` chunked form used by
  ``kenchi.train``), the same formats as Phase 1. Log lines are also appended to
  ``run_<name>.jsonl`` and uploaded to the relay periodically.
- **Stability**: spike = train loss more than 0.3 nats above its EMA (after
  warm-up), counted like Phase 1; NaN/inf loss stops the run (no new
  checkpoints), reported as ``diverged`` with a non-zero exit.

Relative paths in the config (``data_base_dir``, ``ckpt_dir``, ``val_path``,
``resume``, ``init_from``) resolve against the repo root.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from kenchi import ckpt as kckpt
from kenchi import data as kdata
from kenchi import flops as kflops
from kenchi import metrics as kmetrics
from kenchi import shard_stream as kss
from kenchi import tokenizer_core
from kenchi import train as ktrain
from kenchi.model import KenchiConfig
from kenchi.shard_stream import PhaseSpec, ShardStream, SourceSpec
from kenchi.train import _jsonable, _param_groups, build_model, resolve_device

REPO_ROOT = Path(__file__).resolve().parents[1]
T_PROCESS_START = time.perf_counter()

log = logging.getLogger("kenchi.pretrain")

SPIKE_NAT_ABOVE_EMA = 0.3
SPIKE_EMA_DECAY = 0.9
LOG_ALWAYS_STEP_1 = True


# ------------------------------------------------------------------ config

@dataclass
class PretrainConfig:
    """Config for the Phase 2 trainer (see module docstring for the fields)."""

    name: str = "run"
    model: KenchiConfig = field(default_factory=KenchiConfig)
    total_tokens: int = 300_000_000
    warmup_tokens: int = 0
    peak_lr: float = 1.5e-3
    final_lr_frac: float = 0.05
    lr_shape: str = "cosine"
    rows_per_step: int = 128
    seed: int = 0
    device: str = "auto"
    compile: bool = False
    compile_mode: Optional[str] = None
    amp: str = "bf16"
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    time_budget_s: Optional[float] = None
    time_budget_measure_after_s: float = 120.0  # see TimeBudgetReplanner
    stop_after_s: Optional[float] = None        # stop + save a resume checkpoint this long after the first step
    resume_loss_margin: float = 0.5             # resume check: abort if resumed loss is this far above the saved one
    final_resume: bool = False                  # also upload an fp32 resume checkpoint at the end
    log_every: int = 10
    val_every_tokens: int = 20_000_000
    val_batch_rows: int = 8
    val_path: Optional[str] = "data/val.bin"
    val_n_tokens: int = 1_048_576
    # D: validation runs on rank 0 only (other ranks wait at their next all-reduce).
    ckpt_every_s: float = 1500.0     # periodic "resume" checkpoints (bf16)
    ckpt_keep_n: int = 2
    ckpt_dir: str = "ckpts"
    log_upload_every_s: float = 180.0
    final_upload_max_s: float = 2700.0
    accum_steps: int = 1
    sources: Dict[str, Any] = field(default_factory=dict)
    phases: List[Any] = field(default_factory=list)
    data_base_dir: str = "."
    resume: Optional[str] = None
    init_from: Optional[str] = None
    num_threads: Optional[int] = None
    relay_base: Optional[str] = None
    relay_token: Optional[str] = None
    # the job's one run id (env KENCHI_RUN_ID on the node): embedded in
    # checkpoint names + metadata; "auto-resume" picks the newest
    # resume_<run_id>_* checkpoint the relay holds for it.
    run_id: Optional[str] = None


def _resolve_path(p: Optional[str], default: Optional[str] = None) -> Optional[str]:
    if not p:
        return default
    path = Path(p).expanduser()
    return str(path if path.is_absolute() else REPO_ROOT / path)


def load_pretrain_config(path: Union[str, Path], overrides: Sequence[str] = ()) -> PretrainConfig:
    """Load a phase-2 config JSON (with the ``KenchiConfig`` dict in ``model``)."""
    raw = json.loads(Path(path).read_text())
    for ov in overrides:
        k, _, v = ov.partition("=")
        keys = [kk for kk in k.split(".") if kk]
        cur = raw
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        if keys:
            # values are JSON when possible, otherwise a plain string (train.py)
            if v:
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
            else:
                v = None
            cur[keys[-1]] = v
    model = KenchiConfig.from_dict(raw.get("model") or {})

    def _g(key, default):
        v = raw.get(key)
        return default if v is None else v

    cfg = PretrainConfig(
        name=str(_g("name", "run")),
        model=model,
        total_tokens=int(_g("total_tokens", 300_000_000)),
        warmup_tokens=int(_g("warmup_tokens", 0)),
        peak_lr=float(_g("peak_lr", 1.5e-3)),
        final_lr_frac=float(_g("final_lr_frac", 0.05)),
        lr_shape=str(_g("lr_shape", "cosine")),
        rows_per_step=int(_g("rows_per_step", 128)),
        seed=int(_g("seed", 0)),
        device=str(_g("device", "auto")),
        compile=bool(_g("compile", False)),
        compile_mode=raw.get("compile_mode"),
        amp=str(_g("amp", "bf16")),
        grad_clip=float(_g("grad_clip", 1.0)),
        weight_decay=float(_g("weight_decay", 0.1)),
        time_budget_s=_g("time_budget_s", None),
        time_budget_measure_after_s=float(_g("time_budget_measure_after_s", 120.0)),
        stop_after_s=(float(raw["stop_after_s"]) if raw.get("stop_after_s") else None),
        resume_loss_margin=float(_g("resume_loss_margin", 0.5)),
        final_resume=bool(_g("final_resume", False)),
        log_every=int(_g("log_every", 10)),
        val_every_tokens=int(_g("val_every_tokens", 20_000_000)),
        val_batch_rows=int(_g("val_batch_rows", 8)),
        val_path=(_resolve_path(raw["val_path"]) if raw.get("val_path") else
                   (_resolve_path("data/val.bin") if "val_path" not in raw else None)),
        val_n_tokens=int(_g("val_n_tokens", 1_048_576)),
        ckpt_every_s=float(_g("ckpt_every_s", 1500.0)),
        ckpt_keep_n=int(_g("ckpt_keep_n", 2)),
        ckpt_dir=_resolve_path(raw.get("ckpt_dir"), "ckpts"),
        log_upload_every_s=float(_g("log_upload_every_s", 180.0)),
        final_upload_max_s=float(_g("final_upload_max_s", 2700.0)),
        accum_steps=int(_g("accum_steps", 1)),
        sources=dict(raw.get("sources") or {}),
        phases=list(raw.get("phases") or []),
        data_base_dir=_resolve_path(raw.get("data_base_dir"), "."),
        resume=_resolve_path(raw.get("resume")),
        init_from=_resolve_path(raw.get("init_from")),
        num_threads=raw.get("num_threads"),
        relay_base=raw.get("relay_base"),
        relay_token=raw.get("relay_token") or os.environ.get("KENCHI_RELAY_TOKEN"),
    )
    if cfg.time_budget_s is not None:
        cfg.time_budget_s = float(cfg.time_budget_s) or None   # 0 means "no time budget"
    return cfg


def _cfg_to_dict(cfg: PretrainConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    d["model"] = cfg.model.to_dict()
    return d


# ------------------------------------------------------------------ schedule

class TokenSchedule:
    """LR as a function of planned progress.

    ``progress(t) = anchor_p + (t - anchor_tokens) / total`` when anchored,
    else ``t / total``.  ``lr(t) = peak * decay(progress(t))`` where decay is:

    - warm-up: linear in progress up to ``warmup_tokens / total`` of the plan;
    - then a cosine (or linear) decay from peak to
      ``peak * final_lr_frac`` over the rest of the plan.

    ``replan(new_total, t)`` re-anchors so the progress fraction is continuous
    at ``t``: the LR at the replan instant is unchanged.
    """

    def __init__(self, peak_lr: float, final_lr_frac: float = 0.05,
                 shape: str = "cosine", warmup_tokens: int = 0,
                 total_tokens: float = 0):
        if shape not in ("cosine", "linear"):
            raise ValueError(f"lr_shape must be 'cosine' or 'linear', got {shape!r}")
        self.peak_lr = float(peak_lr)
        self.final_lr_frac = float(final_lr_frac)
        self.shape = shape
        self.warmup_tokens = int(warmup_tokens or 0)
        self.total_tokens = float(total_tokens or 0)
        self._anchor_p: Optional[float] = None
        self._anchor_tokens: Optional[float] = None

    # -- progress
    def progress(self, tokens_done: float) -> float:
        if self.total_tokens <= 0:
            return 0.0
        if self._anchor_p is not None and self._anchor_tokens is not None:
            p = self._anchor_p + (float(tokens_done) - self._anchor_tokens) / self.total_tokens
        else:
            p = float(tokens_done) / self.total_tokens
        return min(1.0, max(0.0, p))

    def replan(self, new_total_tokens: float, tokens_done: float) -> None:
        """Re-anchor the plan so progress is continuous at ``tokens_done``."""
        old_p = self.progress(tokens_done)
        self.total_tokens = float(new_total_tokens)
        self._anchor_p = old_p
        self._anchor_tokens = float(tokens_done)

    def apply_plan(self, total_tokens: float, anchor_p: Optional[float] = None,
                   anchor_tokens: float = 0.0) -> None:
        """Set the plan and anchors without computing anything: used to apply the
        rank-0 control vector to every rank (A)."""
        self.total_tokens = float(total_tokens)
        self._anchor_p = float(anchor_p) if anchor_p is not None else None
        self._anchor_tokens = float(anchor_tokens) if anchor_p is not None else None

    def lr_for(self, tokens_done: float) -> float:
        p = self.progress(tokens_done)
        wf = (self.warmup_tokens / self.total_tokens) if (
            self.total_tokens > 0 and self.warmup_tokens > 0) else 0.0
        if wf >= 1.0:
            g = p  # warm-up spans the whole plan
        elif wf > 0 and p <= wf:
            g = p / wf
        else:
            q = (p - wf) / max(1e-12, 1.0 - wf)
            dec = 0.5 * (1.0 + math.cos(math.pi * q)) if self.shape == "cosine" else 1.0 - q
            g = (1.0 - self.final_lr_frac) * dec + self.final_lr_frac
        return self.peak_lr * g

    # -- checkpoint state
    def state(self) -> Dict[str, Any]:
        return {"total_tokens": self.total_tokens, "warmup_tokens": self.warmup_tokens,
                "anchor_p": self._anchor_p, "anchor_tokens": self._anchor_tokens}

    def load_state(self, d: Mapping[str, Any]) -> None:
        # The horizon is the current config's total_tokens (the user resumes with
        # the run's target); the saved anchor re-anchors progress into it.
        self.warmup_tokens = int(d.get("warmup_tokens") or self.warmup_tokens)
        if d.get("total_tokens"):
            # the plan in force when the checkpoint was written (a time budget
            # may have re-planned it); keeps LR and phase continuous on resume
            self.total_tokens = float(d["total_tokens"])
        self._anchor_p = d.get("anchor_p")
        self._anchor_tokens = d.get("anchor_tokens")


class TimeBudgetReplanner:
    """Keeps the token schedule ending when the wall-clock budget ends.

    The rate is measured in THIS process only, from a mark taken a short time
    after the first completed optimizer step (so compilation and the tokens
    done before a resume do not count), and includes everything that costs
    wall time (validation, checkpoint pauses).  Every ``replan_every_s`` the
    plan is set to ``tokens_done + rate * remaining_seconds``; LR and phase
    progress are continuous across a replan (``TokenSchedule.replan``).

    ``measure_after_s`` scales the timing: settle = 0.25x, minimum measuring
    window = 0.75x, replan interval = 2.5x (120 s -> 30 s / 90 s / 300 s).
    """

    MEASURE_AFTER_S = 120.0

    def __init__(self, schedule: TokenSchedule, time_budget_s: float,
                 tokens_per_step: float, measure_after_s: Optional[float] = None,
                 now: Optional[Any] = None):
        self.schedule = schedule
        self.time_budget_s = float(time_budget_s)
        self.tokens_per_step = max(1.0, float(tokens_per_step))
        m = float(measure_after_s if measure_after_s is not None else self.MEASURE_AFTER_S)
        self.settle_s = 0.25 * m
        self.min_window_s = 0.75 * m
        self.replan_every_s = 2.5 * m
        self._now = now or time.perf_counter
        self.t_start = float(self._now())
        self.rate: Optional[float] = None
        self._first_step_t: Optional[float] = None
        self._mark: Optional[Tuple[float, float]] = None
        self._last_replan: Optional[float] = None

    def _now_f(self) -> float:
        return float(self._now())

    def elapsed(self, now: Optional[float] = None) -> float:
        return self._now_f() - self.t_start

    def done(self, now: Optional[float] = None) -> bool:
        return self.elapsed(now) >= self.time_budget_s

    def maybe_replan(self, tokens_done: float, steps_in_process: int = 1) -> Optional[Dict[str, Any]]:
        """Replan when due; returns the replan info when it happened, else None."""
        if steps_in_process < 1:
            return None
        now = self._now_f()
        if self._first_step_t is None:
            self._first_step_t = now
        if self._mark is None:
            if now - self._first_step_t >= self.settle_s:
                self._mark = (now, float(tokens_done))
            return None
        dt = now - self._mark[0]
        if dt < self.min_window_s:
            return None
        if self._last_replan is not None and now - self._last_replan < self.replan_every_s:
            return None
        rate = (float(tokens_done) - self._mark[1]) / max(dt, 1e-9)
        if rate <= 0:
            return None
        self.rate = rate
        self._last_replan = now
        remaining_s = max(0.0, self.t_start + self.time_budget_s - now)
        new_total = int(tokens_done + rate * remaining_s)
        new_total = max(new_total, int(tokens_done) + int(self.tokens_per_step))
        old_total = self.schedule.total_tokens
        self.schedule.replan(new_total, tokens_done)
        return {
            "event": "replan",
            "elapsed_s": now - self.t_start,
            "tokens_done": int(tokens_done),
            "old_total_tokens": old_total,
            "new_total_tokens": new_total,
            "estimated_rate_tok_s": rate,
            "remaining_s": remaining_s,
            "progress": self.schedule.progress(tokens_done),
        }


# ------------------------------------------------------------------ stream

def build_stream(cfg: PretrainConfig, rank: int, world_size: int,
                 device: torch.device, accum_steps: int = 1) -> ShardStream:
    sources = {k: SourceSpec.from_dict(k, v) for k, v in (cfg.sources or {}).items()}
    if not sources:
        sources = {"shards": SourceSpec(name="shards", patterns=["data/shards/*.bin"],
                                        weight=1.0)}
    phases = [PhaseSpec.from_dict(p) for p in (cfg.phases or [])]
    if not phases:
        phases = [PhaseSpec(tokens=int(cfg.total_tokens), name="main")]
    T = int(cfg.model.seq_len)
    micro_rows = max(1, math.ceil(cfg.rows_per_step /
                                  max(1, world_size * max(1, int(accum_steps)))))
    return ShardStream(sources=sources, phases=phases, seq_len=T, batch_size=micro_rows,
                       rank=rank, world_size=world_size, device=device, seed=cfg.seed,
                       base_dir=cfg.data_base_dir or ".",
                       total_tokens=int(cfg.total_tokens))


# ------------------------------------------------------------------ trainer

def _rank_world() -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
        try:
            return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        except ValueError:
            pass
    return 0, 1


class _LogUploader:
    """Appends emitted log lines to run_<name>.jsonl and uploads it periodically.
    Rank 0 only (F.5): every rank appending to one file would interleave/corrupt it."""

    def __init__(self, path: Path, remote_name: str, every_s: float, rank: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.remote_name = remote_name
        self.every_s = float(every_s)
        self.rank = int(rank)
        self._last = 0.0

    def note(self, text: str) -> None:
        if self.rank != 0:
            return
        try:
            with open(self.path, "a") as f:
                f.write(text + "\n")
        except OSError as e:  # logging must never crash the run
            log.warning("[pretrain] log file write failed: %s", e)

    def maybe_upload(self) -> None:
        if self.rank != 0:
            return
        now = time.perf_counter()
        if self.every_s > 0 and now - self._last >= self.every_s:
            self._last = now
            kckpt.upload_async(self.path, self.remote_name)
            log.info("[pretrain] queued log upload of %s", self.remote_name)


def _param_checksum(model) -> str:
    """sha256 over (name, dtype, bytes) of every unique parameter -- used to check
    that DDP ranks end a run with identical parameters."""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(str(p.dtype).encode())
        h.update(p.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def run_pretrain(cfg: PretrainConfig) -> Dict[str, Any]:
    """Train per docs/PHASE1_DESIGN.md §4 / Phase 2 spec; return the RESULT_JSON dict.

    Returns a dict with at least ``step``/``tokens``/``losses``/``diverged``;
    ``main`` maps ``diverged`` to a non-zero exit.
    """
    device = resolve_device(cfg.device)
    # (2) the job asserts the tokenizer file's sha256 at start
    tokenizer_sha = tokenizer_core.tokenizer_file_sha256()
    log.info("[pretrain] tokenizer sha256: %s", tokenizer_sha)
    run_id = os.environ.get("KENCHI_RUN_ID") or cfg.run_id or cfg.name
    # process group: torchrun sets RANK/WORLD_SIZE in the environment
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank, world = _rank_world()
    elif os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
        torch.distributed.init_process_group(
            backend="nccl" if device == "cuda" else "gloo")
        rank, world = _rank_world()
    else:
        rank, world = 0, 1

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        # B: use TF32 (see notes; requires Ampere+).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if cfg.num_threads:
        torch.set_num_threads(int(cfg.num_threads))
    elif os.environ.get("OMP_NUM_THREADS"):
        # the entry script exports OMP_NUM_THREADS=8 per rank (see notes)
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    # B: identical init seed on all ranks; per-rank data/RNG seeds after build
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2 ** 32))

    T = int(cfg.model.seq_len)
    if cfg.total_tokens <= 0 and not cfg.time_budget_s:
        raise ValueError("total_tokens must be > 0 (or set time_budget_s)")

    # C: build the bare model, apply torch.compile to it, then wrap in DDP
    compile_model = bool(cfg.compile) and device == "cuda"
    model = build_model(cfg.model, device)
    if compile_model:
        model = torch.compile(model, mode=cfg.compile_mode)
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device == "cuda" else None)
    raw_model = kckpt.unwrap(model)          # Kenchi, without DDP/compile wrappers
    torch.manual_seed(cfg.seed + rank)        # per-rank data/RNG seeds (B)
    random.seed(cfg.seed + rank)
    np.random.seed((cfg.seed + rank) % (2 ** 32))

    train_flops_tok = kflops.train_flops_per_token(cfg.model, T)
    params_info = kflops.count_params(raw_model)

    optimizer = torch.optim.AdamW(_param_groups(raw_model, cfg.weight_decay),
                                  betas=(0.9, 0.95), lr=0.0,
                                  fused=(device == "cuda"))

    schedule = TokenSchedule(cfg.peak_lr, cfg.final_lr_frac, cfg.lr_shape,
                             cfg.warmup_tokens, cfg.total_tokens)
    stream = build_stream(cfg, rank, world, device, accum_steps=cfg.accum_steps)

    ckpt = kckpt.Checkpointer(cfg.ckpt_dir, cfg.name, raw_model, optimizer,
                              keep_n=cfg.ckpt_keep_n,
                              upload=True, relay_base=cfg.relay_base,
                              relay_token=cfg.relay_token, rank=rank,
                              final_upload_max_s=cfg.final_upload_max_s,
                              run_id=run_id, tokenizer_sha256=tokenizer_sha)

    log_uploader = _LogUploader(Path(cfg.ckpt_dir) / f"run_{cfg.name}.jsonl",
                                f"run_{cfg.name}.jsonl", cfg.log_upload_every_s,
                                rank=rank)

    def emit(tag: str, obj: Dict[str, Any]) -> None:
        obj = {**obj, "run": cfg.name}
        text = json.dumps(_jsonable(obj), default=str)
        if rank == 0:
            print(f"{tag} {text}", flush=True)
        log_uploader.note(f"{tag} {text}")

    def val_set():
        if not cfg.val_path or not Path(cfg.val_path).is_file():
            return None
        return kdata.FixedValSet(cfg.val_path, batch_size=cfg.val_batch_rows,
                                  T=T, device=device, n_tokens=cfg.val_n_tokens)

    def validate(vs) -> Optional[float]:
        """Validation loss on the fixed validation shard (D): rank 0 only; the
        other ranks are already waiting at their next all-reduce, so rank 0
        validating alone cannot deadlock a DDP job."""
        if vs is None or rank != 0:
            return None
        was = raw_model.training
        raw_model.eval()
        try:
            return kmetrics.val_loss(raw_model, vs.tokens, cfg=cfg.model,
                                     autocast_ctx=autocast_ctx,
                                     chunk=cfg.val_batch_rows)
        finally:
            if was:
                raw_model.train()

    # ---- resume / init (under DDP rank 0 loads first, then a barrier, then
    #      every rank loads the same local file)
    def ranks_agree(tag: str, checksum: str) -> bool:
        """True when every rank holds bit-identical parameters (48 bits of the
        sha256 compared through an all-reduce)."""
        if world == 1:
            return True
        v = float(int(checksum[:12], 16))
        t = torch.tensor([v, -v], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        ok = bool(t[0] == -t[1])
        if rank == 0:
            print(f"RANKS_AGREE {tag} {ok}", flush=True)
        return ok

    start_step, tokens_done = 0, 0
    base_trained_s = 0.0
    resumed: Optional[Dict[str, Any]] = None
    saved_recent_loss: Optional[float] = None
    if cfg.resume:
        if rank == 0:
            resumed = ckpt.restore(cfg.resume, model=model, optimizer=optimizer,
                                   stream=stream, schedule=schedule, rank=rank)
        if world > 1:
            torch.distributed.barrier()
        if rank != 0:
            resumed = ckpt.restore(cfg.resume, model=model, optimizer=optimizer,
                                   stream=stream, schedule=schedule, rank=rank)
        start_step = int(resumed.get("step", 0))
        tokens_done = int(resumed.get("tokens", 0))
        base_trained_s = float(resumed.get("trained_s", 0.0) or 0.0)
        saved_recent_loss = resumed.get("recent_loss")
        # refuse to resume from a checkpoint whose tokenizer differs
        ck_tok_sha = resumed.get("tokenizer_sha256")
        expected_tok_sha = os.environ.get("KENCHI_TOKENIZER_SHA256") or tokenizer_sha
        if expected_tok_sha and ck_tok_sha and ck_tok_sha != expected_tok_sha:
            print(f"TOKENIZER_MISMATCH: checkpoint tokenizer sha256={ck_tok_sha} != "
                  f"expected {expected_tok_sha}; refusing to resume", flush=True)
            raise SystemExit(2)
        csum = _param_checksum(raw_model)
        print(f"PARAM_CHECKSUM {rank} {csum} phase=resumed "
              f"step={start_step} tokens={tokens_done}", flush=True)
        agree = ranks_agree("resumed", csum)
        # resume checks: every one must hold or the run stops before training
        problems: List[str] = []
        lr_now = schedule.lr_for(tokens_done)
        lr_saved = resumed.get("lr")
        opt_steps = [int(v["step"].item()) for v in optimizer.state_dict()["state"].values()
                     if isinstance(v, dict) and isinstance(v.get("step"), torch.Tensor)]
        sq_norm = max([float(v["exp_avg_sq"].float().norm()) for v in
                       optimizer.state_dict()["state"].values()
                       if isinstance(v, dict) and isinstance(v.get("exp_avg_sq"), torch.Tensor)]
                      or [0.0])
        s_state = stream.state()
        if start_step <= 0 or tokens_done <= 0:
            problems.append(f"step/tokens not positive ({start_step}/{tokens_done})")
        if lr_saved is not None and abs(lr_now - float(lr_saved)) > 1e-6 * max(abs(float(lr_saved)), 1e-12):
            problems.append(f"LR not continuous (saved {lr_saved:.6e}, now {lr_now:.6e})")
        if opt_steps and (min(opt_steps) != start_step or max(opt_steps) != start_step):
            problems.append(f"optimizer step {min(opt_steps)}..{max(opt_steps)} != {start_step}")
        if not opt_steps or sq_norm <= 0:
            problems.append("optimizer state empty")
        if int(s_state.get("step", 0)) <= 0:
            problems.append("data stream position not restored")
        if resumed.get("run_id") and resumed.get("run_id") != run_id:
            problems.append(f"run id {resumed.get('run_id')} != {run_id}")
        if not agree:
            problems.append("parameters differ between ranks after loading")
        if rank == 0:
            print(f"RESUME_CHECK step={start_step} tokens={tokens_done} "
                  f"trained_s={base_trained_s:.1f} lr_saved={lr_saved} lr_now={lr_now:.6e} "
                  f"optimizer_step={opt_steps[0] if opt_steps else None} "
                  f"optimizer_sq_norm={sq_norm:.4g} stream_step={s_state.get('step')} "
                  f"plan_total={schedule.total_tokens:.0f} progress={schedule.progress(tokens_done):.5f} "
                  f"saved_recent_loss={saved_recent_loss} tokenizer_sha256={ck_tok_sha} "
                  f"run_id={resumed.get('run_id')}", flush=True)
            print(f"RESUME_CHECK stream_cursors={json.dumps(s_state.get('cursors'))}", flush=True)
            print("RESUME_CHECK " + ("PASS (load)" if not problems else "FAIL: " + "; ".join(problems)),
                  flush=True)
        if problems:
            raise SystemExit(3)
    elif cfg.init_from:
        kckpt.init_from(cfg.init_from, raw_model)
        log.info("[pretrain] initialized weights from %s", cfg.init_from)

    _proc_t0 = time.perf_counter()

    def _trained_s() -> float:
        """Accumulated training seconds: the restored base plus this process."""
        return base_trained_s + (time.perf_counter() - _proc_t0)

    # ---- training setup
    model.train()
    fwd = model
    amp_on = cfg.amp == "bf16" and device in ("cuda", "cpu")
    autocast_ctx = (torch.autocast(device_type=device, dtype=torch.bfloat16)
                    if amp_on else nullcontext())

    micro_rows = stream.batch_size
    accum_steps = max(1, int(cfg.accum_steps))
    n_micro = accum_steps
    rows_per_step = micro_rows * world * n_micro
    tokens_per_step = rows_per_step * T
    cfg_dict = _cfg_to_dict(cfg)
    cfg_dict.pop("relay_token", None)               # never written to a file
    cfg_dict["tokenizer_sha256"] = tokenizer_sha
    cfg_dict["run_id"] = run_id

    # the time budget across restarts: ACCUMULATED training seconds (restored
    # from the resume checkpoint), not the process uptime
    time_budget_remaining = (max(0.0, float(cfg.time_budget_s) - base_trained_s)
                             if cfg.time_budget_s else None)
    replanner = (TimeBudgetReplanner(schedule, time_budget_remaining, tokens_per_step,
                                     measure_after_s=cfg.time_budget_measure_after_s)
                 if cfg.time_budget_s else None)

    vs = val_set()
    if vs is not None and rank == 0:
        log.info("[pretrain] validation on %d tokens of %s", vs.tokens.numel(), cfg.val_path)

    # ---- loop state
    step, tokens = start_step, tokens_done
    losses: List[float] = []
    ema_loss: Optional[float] = None
    spikes = 0
    max_gn = 0.0
    diverged = False
    truncated = False
    stop_reason = 0          # 0 plan complete, 1 time budget, 2 SIGTERM, 3 stop_after_s, 4 resume check
    stop_ckpt_ok = False
    resume_fail = False
    resume_loss_checked = resumed is None
    replan_events: List[Dict[str, Any]] = []
    val_curve: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    t_first_step: Optional[float] = None
    tp0, tok0 = t_start, 0
    t_eval_total = 0.0
    last_resume = t_start
    last_val_tokens = tokens
    stream.advance_phase_by_progress(schedule.progress(tokens), tokens)
    ckpt.note_progress(step, tokens)

    def ckpt_extra() -> Dict[str, Any]:
        recent = losses[-30:]
        return {"lr": schedule.lr_for(tokens),
                "recent_loss": (sum(recent) / len(recent)) if recent else saved_recent_loss}

    emit("START_JSON", {
        "run": cfg.name,
        "run_id": run_id,
        "params": params_info,
        "macs_per_token": kflops.forward_macs_per_token(cfg.model, T),
        "train_flops_per_token": train_flops_tok,
        "total_tokens": cfg.total_tokens,
        "time_budget_s": cfg.time_budget_s,
        "time_budget_remaining_s": time_budget_remaining,
        "stop_after_s": cfg.stop_after_s,
        "device": device,
        "world_size": world,
        "torch_version": torch.__version__,
        "amp": cfg.amp,
        "compile": compile_model,
        "rows_per_step": rows_per_step,
        "accum_steps": accum_steps,
        "T": T,
        "tokenizer_sha256": tokenizer_sha,
        "sources": {k: {"weight": s.weight, "shards": len(s.shard_paths(stream.base_dir))}
                    for k, s in stream.sources.items()},
        "phases": [{"name": p.name or f"phase{i}", "tokens": p.tokens, "fraction": p.fraction,
                    "weights": p.weights} for i, p in enumerate(stream.phases)],
        "resumed_from": cfg.resume,
        "resumed_step": start_step,
        "resumed_tokens": tokens_done,
        "init_from": cfg.init_from,
    })

    param_checksum = _param_checksum(raw_model)
    print(f"PARAM_CHECKSUM {rank} {param_checksum} phase=init", flush=True)
    if not ranks_agree("init", param_checksum):
        print("RANK_PARAM_MISMATCH at start; stopping", flush=True)
        raise SystemExit(5)

    while True:
        now = time.perf_counter()
        # ---- rank 0 is the only decision maker; the control vector
        # [stop, planned_total, anchor_p, anchor_tokens, stop_reason] is broadcast
        # every step and applied by every rank before computing the LR.
        # (NCCL groups need CUDA tensors.)
        ctrl = torch.zeros(5, dtype=torch.float64, device=device)
        if rank == 0:
            reason = 0
            if replanner is not None and replanner.done():
                reason = 1
            if (cfg.stop_after_s and t_first_step is not None
                    and now - t_first_step >= float(cfg.stop_after_s)):
                reason = 3
            if ckpt.terminate_requested():
                reason = 2
            if resume_fail:
                reason = 4
            if reason:
                ctrl[0], ctrl[4] = 1.0, float(reason)
            info = (replanner.maybe_replan(tokens, step - start_step)
                    if (replanner is not None and not reason) else None)
            if info:
                replan_events.append(info)
                emit("REPLAN_JSON", {**info, "lr": schedule.lr_for(tokens)})
            ctrl[1] = schedule.total_tokens
            ctrl[2] = (schedule._anchor_p if schedule._anchor_p is not None else -1.0)
            ctrl[3] = (schedule._anchor_tokens if schedule._anchor_tokens is not None
                       else 0.0)
        if world > 1:
            # SIGTERM may arrive on any rank; combine with an all-reduce MAX
            t = torch.tensor([1.0 if ckpt.terminate_requested() else 0.0], device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            if rank == 0 and t[0] > 0 and ctrl[0] == 0:
                ctrl[0], ctrl[4] = 1.0, 2.0
            torch.distributed.broadcast(ctrl, src=0)
        if ctrl[0] > 0:
            stop_reason = int(ctrl[4])
            if stop_reason in (2, 3) and rank == 0:
                # stop at a step boundary: save a resume checkpoint and upload it
                # (verified) before exiting
                p, stop_ckpt_ok = ckpt.save_resume(
                    step, tokens, schedule=schedule, stream=stream, config=cfg_dict,
                    wait=True, trained_s=_trained_s(), tokenizer_sha256=tokenizer_sha,
                    extra=ckpt_extra())
                print(f"STOP_CHECKPOINT reason={stop_reason} step={step} tokens={tokens} "
                      f"file={p.name if p else None} delivered={stop_ckpt_ok}", flush=True)
            truncated = stop_reason == 1
            break
        if ctrl[1] > 0:     # apply the plan; identical on every rank
            schedule.apply_plan(float(ctrl[1]), float(ctrl[2]) if ctrl[2] >= 0 else None,
                                float(ctrl[3]))
        plan_total = schedule.total_tokens if replanner is not None else cfg.total_tokens
        if tokens >= plan_total:
            break
        stream.advance_phase_by_progress(schedule.progress(tokens), tokens)

        # ---- learning rate for this step (token schedule)
        lr = schedule.lr_for(tokens)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ---- micro steps (gradient accumulation)
        micro_losses: List[float] = []
        last_out: Dict[str, Any] = {}
        for _ in range(n_micro):
            x, y = stream.next_batch()
            if rank == 0 and step == start_step and not micro_losses:
                # decode the first rows of this process so the log shows real text
                tok = tokenizer_core.load_tokenizer()
                for i in range(min(3, int(x.shape[0]))):
                    text = tok.decode([int(t) for t in x[i, :200].detach().cpu().tolist()])
                    print(f"TOKENIZER_SAMPLE {i} {text[:160]!r}", flush=True)
            with autocast_ctx:
                out = fwd(x, targets=y, return_logits=False)
            last_out = out
            # collective divergence check BEFORE backward: with DDP every rank
            # must call backward the same number of times
            finite = 1.0 if torch.isfinite(out["loss"]) else 0.0
            if world > 1:
                t = torch.tensor([finite], device=device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
                finite = float(t[0])
            if finite == 0:
                diverged = True
                break
            (out["loss"] / n_micro).backward()
            micro_losses.append(float(out["loss"].detach()))
        if diverged:
            break
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens += tokens_per_step
        losses.append(sum(micro_losses) / len(micro_losses))
        now = time.perf_counter()
        if t_first_step is None:
            t_first_step = now
            tp0, tok0 = now, tokens      # throughput is measured after the first step (compile)
        if cfg.model.jepa == "ema" and hasattr(raw_model, "update_ema"):
            raw_model.update_ema()
        ckpt.note_progress(step, tokens)

        lv = losses[-1]
        # spike = loss more than 0.3 nats above its EMA (after warm-up), as in Phase 1
        if (ema_loss is not None and tokens > cfg.warmup_tokens
                and lv > ema_loss + SPIKE_NAT_ABOVE_EMA):
            spikes += 1
        if ema_loss is None:
            ema_loss = lv
        else:
            ema_loss = SPIKE_EMA_DECAY * ema_loss + (1 - SPIKE_EMA_DECAY) * lv
        max_gn = max(max_gn, gn)

        # ---- resume check on the loss: the first resumed steps must not be far
        # above the loss saved in the checkpoint
        if not resume_loss_checked and rank == 0 and len(losses) >= 30:
            resume_loss_checked = True
            mean30 = sum(losses[:30]) / 30.0
            if saved_recent_loss is None:
                print(f"RESUME_CHECK loss mean_first30={mean30:.4f} (no saved loss to compare)",
                      flush=True)
            else:
                ok = mean30 <= float(saved_recent_loss) + cfg.resume_loss_margin
                print(f"RESUME_CHECK loss mean_first30={mean30:.4f} saved_recent={float(saved_recent_loss):.4f} "
                      f"delta={mean30 - float(saved_recent_loss):+.4f} "
                      + ("PASS (loss)" if ok else "FAIL: resumed loss too high"), flush=True)
                resume_fail = not ok

        # ---- periodic resume checkpoints (weights first, then weights + optimizer)
        if now - last_resume >= cfg.ckpt_every_s and not resume_fail:
            last_resume = now
            p, _ = ckpt.save_resume(step, tokens, schedule=schedule, stream=stream,
                                    config=cfg_dict, wait=False, trained_s=_trained_s(),
                                    tokenizer_sha256=tokenizer_sha, extra=ckpt_extra())
            if p is not None and rank == 0:
                print(f"CKPT_QUEUED {p.name} step={step} tokens={tokens}", flush=True)

        # ---- logging
        if step % cfg.log_every == 0 or step - start_step == 1:
            dt = now - tp0 - t_eval_total
            rate = ((tokens - tok0) / dt) if dt > 1e-6 else 0.0
            plan_total = schedule.total_tokens if replanner is not None else cfg.total_tokens
            eta = ((plan_total - tokens) / rate) if rate > 0 else None
            aux = {k: float(v.detach()) for k, v in last_out.items()
                   if k.startswith("loss_") and isinstance(v, torch.Tensor)
                   and v.numel() == 1}
            emit("LOG_JSON", {
                "step": step, "tokens": tokens, "loss": lv,
                "aux": aux,
                "lr": schedule.lr_for(tokens), "grad_norm": gn,
                "tok_per_s": rate, "elapsed": now - t_start,
                "trained_s": _trained_s(),
                "progress": schedule.progress(tokens),
                "phase": stream.current_phase_name(),
                "gpu_mem_mb": (float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
                               if device == "cuda" else None),
                "eta_s": eta,
            })
        log_uploader.maybe_upload()

        # ---- validation
        # (one early validation on every start, so a start that cannot validate
        # fails in its first minute, not after an hour)
        if vs is not None and (tokens - last_val_tokens >= cfg.val_every_tokens
                               or step - start_step == 30):
            t_v0 = time.perf_counter()
            vl = validate(vs)
            t_eval_total += time.perf_counter() - t_v0
            if step - start_step != 30:
                last_val_tokens = tokens
            if vl is not None:
                val_curve.append({"step": step, "tokens": tokens, "val_loss": vl})
                emit("VAL_JSON", {"step": step, "tokens": tokens, "val_loss": vl,
                                  "seconds": time.perf_counter() - t_v0})

    # ---- end of training
    diverged_note = ("diverged: loss became NaN/inf at step %d (tokens=%d); "
                     "no new checkpoints saved" % (step, tokens)) if diverged else None
    finished = (not diverged) and stop_reason in (0, 1)
    if diverged and rank == 0:
        print(f"DIVERGED {diverged_note}", flush=True)

    elapsed = time.perf_counter() - t_start
    rate = (tokens - tok0) / max(time.perf_counter() - tp0 - t_eval_total, 1e-9) if t_first_step else 0.0
    val_loss_final = None
    if finished and vs is not None:
        val_loss_final = validate(vs)

    param_checksum = _param_checksum(raw_model)
    print(f"PARAM_CHECKSUM {rank} {param_checksum} phase=end step={step} tokens={tokens}",
          flush=True)
    ranks_equal_end = ranks_agree("end", param_checksum)

    result: Dict[str, Any] = {
        "run": cfg.name,
        "run_id": run_id,
        "config": cfg_dict,
        "device": device,
        "steps_run": step,
        "tokens": tokens,
        "total_tokens": cfg.total_tokens,
        "plan_total_tokens": schedule.total_tokens,
        "progress": schedule.progress(tokens),
        "phase_at_end": stream.current_phase_name(),
        "phase_events": [e for e in stream.events if e.get("event") == "phase_switch"],
        "wraps": sum(1 for e in stream.events if e.get("event") == "wrap"),
        "losses": losses[-50:],
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_recent": (sum(losses[-50:]) / len(losses[-50:])) if losses else None,
        "elapsed_s": elapsed,
        "tok_per_s": rate,
        "val_loss_final": val_loss_final,
        "val_curve": val_curve,
        "stability": {"spikes": spikes, "max_grad_norm": max_gn,
                      "diverged": diverged, "diverged_note": diverged_note},
        "diverged": diverged,
        "truncated": truncated,
        "stop_reason": {0: "plan_complete", 1: "time_budget", 2: "sigterm", 3: "stop_after",
                        4: "resume_check_failed"}.get(stop_reason, str(stop_reason)),
        "stop_checkpoint_delivered": stop_ckpt_ok,
        "resume_check_failed": stop_reason == 4,
        "finished": finished,
        "trained_s": _trained_s(),
        "replan_events": replan_events[-20:],
        "param_checksum": param_checksum,
        "ranks_equal_end": ranks_equal_end,
        "rank": rank,
        "world_size": world,
    }
    emit("RESULT_JSON", result)
    if finished and rank == 0:
        # final artifacts, in order: weights, config, tokenizer, log (then, only
        # when asked for, an fp32 resume checkpoint).  Each is verified by
        # sha256 + size on the relay; bounded retries; one line per artifact.
        delivered = ckpt.final_uploads(cfg.ckpt_dir, cfg_dict, schedule=schedule, stream=stream,
                                       log_path=log_uploader.path,
                                       tokenizer_path=tokenizer_core._find_model_path(),
                                       with_resume=cfg.final_resume)
        result["final_uploads"] = delivered
        print("FINAL_UPLOADS_JSON " + json.dumps(delivered), flush=True)
    return result


# ------------------------------------------------------------------ CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kenchi.pretrain",
                                 description="Phase 2 pretraining trainer")
    ap.add_argument("--config", required=True, help="phase 2 config JSON path")
    ap.add_argument("--override", action="append", default=[],
                    help="key=value overrides; dotted keys like model.d_model; values are JSON")
    ap.add_argument("--resume", default=None, help="relay name or local path of a full ckpt")
    ap.add_argument("--init_from", default=None, help="light ckpt (weights only)")
    ap.add_argument("--time_budget_s", type=float, default=None)
    ap.add_argument("--stop_after_s", type=float, default=None,
                    help="stop this many seconds after the first completed step, save + upload a "
                         "resume checkpoint and exit 0 (the sanity job)")
    args = ap.parse_args(argv)

    cfg = load_pretrain_config(args.config, args.override)
    if args.resume:
        cfg.resume = args.resume
    if args.init_from:
        cfg.init_from = args.init_from
    if args.time_budget_s is not None:
        cfg.time_budget_s = args.time_budget_s or None
    if args.stop_after_s is not None:
        cfg.stop_after_s = args.stop_after_s or None

    result = run_pretrain(cfg)
    if result.get("diverged"):
        print("pretrain: exiting non-zero due to divergence", file=sys.stderr)
        return 2
    if result.get("resume_check_failed"):
        print("pretrain: exiting non-zero: resume check failed", file=sys.stderr)
        return 3
    if result.get("stop_reason") in ("stop_after", "sigterm") and result.get("rank") == 0 \
            and not result.get("stop_checkpoint_delivered"):
        print("pretrain: exiting non-zero: the stop checkpoint was not delivered", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.exit(main())
