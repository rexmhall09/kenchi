"""Trainer for Phase 1 (src/kenchi/train.py).

CLI:
    python -m kenchi.train --config configs/phase1/<name>.json [--override key=value ...]
                           [--max_seconds S] [--dry_run_steps N]

Implements docs/PHASE1_DESIGN.md §4: AdamW param groups (beta=(0.9, 0.95); weight
decay 0.1 on >=2-D non-embedding weights; fused on CUDA), schedule (linear warm-up
to the peak lr, cosine to final_lr_frac of the peak), JSON-line logging --
`LOG_JSON`, `VAL_JSON` and `RESULT_JSON` lines on stdout, each followed by one JSON
object including the run name -- every LOG_INTERVAL=50 steps, validation loss at
N_VAL_POINTS=12 evenly spaced points, spike detection (a logged train loss more than
0.3 nats above its EMA after warm-up) and NaN/inf abort (reported as a divergence),
throughput excluding compile warm-up and eval time, and peak GPU memory. Works on
cpu, cuda and mps; torch.compile only when compile=true and device is cuda. With
loop sampling the sampled n is drawn in Python per step and passed as n_loops.

Relative paths in the config (data_dir, val_path) resolve against the repo root.

In stub mode (kenchi.model not landed yet, tests only) kenchi.model/flops are
replaced by tests/_stub_model.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import gzip
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from kenchi import data as kdata
from kenchi import flops as kflops
from kenchi import metrics as kmetrics
from kenchi import model as kmodel
from kenchi import recall_task as krecall
from kenchi.model import KenchiConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
T_PROCESS_START = time.perf_counter()   # process start, for first_step_s (§G4)
VAL_TOKENS_PER_BYTE = 0.216483  # measured tokens-per-byte constant of the validation shard (§6)
LOG_INTERVAL = 50
N_VAL_POINTS = 12
SPIKE_NAT_ABOVE_EMA = 0.3
SPIKE_EMA_DECAY = 0.9
AUX_KEYS = ("loss_lm", "loss_jepa", "loss_u", "loss_mtp")


# ------------------------------------------------------------------ config


@dataclass
class TrainConfig:
    name: str = "run"
    model: Any = None
    task: str = "lm"
    flop_budget: Optional[float] = None
    token_budget: Optional[float] = None
    batch_size: int = 64
    lr: float = 1.5e-3
    warmup_frac: float = 0.02
    final_lr_frac: float = 0.05
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    p_syn: float = 0.05  # fraction of training rows replaced with recall docs (default)
    signal_n_tokens: int = 524288  # calibration + signal table run on the first N val tokens
    num_threads: Optional[int] = None      # torch.set_num_threads(N) if set
    entity_mask_path: Optional[str] = None  # path to a reduced-vocab entity_mask.npy
    seed: int = 0
    data_dir: Optional[str] = "data/shards"
    n_train_shards: int = 8
    val_path: Optional[str] = "data/val.bin"
    val_n_tokens: int = 2_097_152
    eval_n_recall_docs: int = 8
    eval_recall_n_values: List[int] = field(default_factory=lambda: [8, 32, 64, 128])
    compile: bool = False
    amp: str = "bf16"
    device: str = "auto"
    max_seconds: Optional[float] = None
    dry_run_steps: Optional[int] = None
    compile_mode: Optional[str] = None   # torch.compile mode; null -> torch default
    eval_batch_rows: int = 16            # rows per forward in every evaluation (§G2)


def _parse_override(s: str):
    k, _, v = s.partition("=")
    try:
        v = json.loads(v)
    except Exception:
        pass
    return k, v


def _set_nested(obj, dotted: str, value):
    keys = [k for k in dotted.split(".") if k]
    if not keys:
        return
    cur = obj
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def load_config(path, overrides: Sequence[str] = ()) -> TrainConfig:
    raw = json.loads(Path(path).read_text())
    for ov in overrides:
        k, v = _parse_override(ov)
        _set_nested(raw, k, v)
    model_dict = raw.get("model") or {}
    valid = {f.name for f in dc_fields(KenchiConfig)}
    model = KenchiConfig(**{k: v for k, v in model_dict.items() if k in valid})
    tcfg = TrainConfig(
        name=raw.get("name", "run"),
        model=model,
        task=raw.get("task", "lm"),
        flop_budget=raw.get("flop_budget"),
        token_budget=raw.get("token_budget"),
        batch_size=int(raw.get("batch_size", 64)),
        lr=float(raw.get("lr", 1.5e-3)),
        warmup_frac=float(raw.get("warmup_frac", 0.02)),
        final_lr_frac=float(raw.get("final_lr_frac", 0.05)),
        weight_decay=float(raw.get("weight_decay", 0.1)),
        grad_clip=float(raw.get("grad_clip", 1.0)),
        p_syn=float(raw.get("p_syn", 0.05)),
        seed=int(raw.get("seed", 0)),
        data_dir=raw.get("data_dir", "data/shards"),
        n_train_shards=int(raw.get("n_train_shards", 8)),
        val_path=raw.get("val_path", "data/val.bin"),
        val_n_tokens=int(raw.get("val_n_tokens", 2_097_152)),
        eval_n_recall_docs=int(raw.get("eval_n_recall_docs", 8)),
        eval_recall_n_values=[int(x) for x in raw.get("eval_recall_n_values", [8, 32, 64, 128])],
        compile=bool(raw.get("compile", False)),
        amp=raw.get("amp", "bf16"),
        device=raw.get("device", "auto"),
        max_seconds=raw.get("max_seconds"),
        dry_run_steps=raw.get("dry_run_steps"),
        signal_n_tokens=int(raw.get("signal_n_tokens", 524288)),
        num_threads=raw.get("num_threads"),
        entity_mask_path=raw.get("entity_mask_path"),
        compile_mode=raw.get("compile_mode"),
        eval_batch_rows=int(raw.get("eval_batch_rows", 16)),
    )
    if tcfg.task not in ("lm", "recall"):
        raise ValueError(f"task must be 'lm' or 'recall', got {tcfg.task!r}")
    if tcfg.flop_budget is None and tcfg.token_budget is None:
        raise ValueError("the config needs flop_budget or token_budget")
    if tcfg.task == "lm" and not tcfg.val_path:
        raise ValueError("task=lm needs val_path")
    return tcfg


def resolve_device(spec: str) -> str:
    if spec not in ("auto", None, ""):
        return spec
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def build_model(cfg_model, device):
    """The Kenchi model class from kenchi.model (§1.7)."""
    for name in ("Kenchi", "KenchiModel", "Model"):
        cls = getattr(kmodel, name, None)
        if cls is not None:
            return cls(cfg_model).to(device)
    raise RuntimeError("kenchi.model defines none of 'Kenchi', 'KenchiModel', 'Model'")


def _macs_per_token(flops_mod, cfg_model, model, T) -> int:
    last = None
    for args in ((cfg_model, T), (model, T)):
        try:
            return int(flops_mod.forward_macs_per_token(*args))
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"cannot call flops.forward_macs_per_token(cfg, T): {last}")


def _train_flops_per_token(flops_mod, cfg_model, model, T) -> int:
    last = None
    for args in ((cfg_model, T), (model, T)):
        try:
            return int(flops_mod.train_flops_per_token(*args))
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"cannot call flops.train_flops_per_token(cfg, T): {last}")


def _resolve_path(p: Optional[str]) -> Optional[Path]:
    if p is None:
        return None
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _param_info(model, flops_mod) -> Dict[str, int]:
    info = None
    if hasattr(flops_mod, "count_params"):
        try:
            info = flops_mod.count_params(model)
        except Exception:
            info = None
    if isinstance(info, dict):
        emb = int(info.get("embedding", 0) or 0)
        body = int(info.get("body", 0) or 0)
        heads = int(info.get("heads", 0) or 0)
        total = int(info.get("total", 0) or 0) or emb + body + heads
        return {"embedding": emb, "body": body, "heads": heads, "total": total}
    emb = body = heads = 0
    for name, p in model.named_parameters():
        n = p.numel()
        if "embed" in name or "lm_head" in name or "unembed" in name:
            emb += n
        elif "head" in name:
            heads += n
        else:
            body += n
    return {"embedding": emb, "body": body, "heads": heads,
            "total": emb + body + heads}


def _param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in name or "lm_head" in name or "unembed" in name:
            no_decay.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


# ------------------------------------------------------------------ helpers


def _validation_loss(model, cfg, val_tokens, val_recall_tokens, autocast_ctx) -> float:
    was_training = getattr(model, "training", True)
    if hasattr(model, "eval"):
        model.eval()
    try:
        if cfg.task == "lm":
            vl = kmetrics.val_loss(model, val_tokens, cfg=cfg.model,
                                  autocast_ctx=autocast_ctx,
                                  chunk=cfg.eval_batch_rows)
        else:
            vl = kmetrics.val_loss(model, val_recall_tokens, cfg=cfg.model,
                                  autocast_ctx=autocast_ctx,
                                  chunk=cfg.eval_batch_rows)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return vl


def _mix_recall_batch(x, y, recall_gen, p_syn, pick_gen):
    """Replace a fraction `p_syn` of the rows of (x, y) with recall documents.

    Picks the rows first (seeded generator owned by the trainer, not the global
    RNG), then generates only that many recall rows. Returns
    (tokens, targets, loss_mask): recall rows supervise the two answer-predicting
    positions of every query; natural rows supervise everything (§4, §5).
    """
    B = x.shape[0]
    sel = (torch.rand(B, generator=pick_gen) < float(p_syn)).to(x.device)
    idx = torch.nonzero(sel, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return x, y, None
    rec = recall_gen.next_batch(n_rows=int(idx.numel()))
    tokens = x.clone()
    targets = y.clone()
    loss_mask = torch.ones_like(x, dtype=torch.bool)
    tokens[idx] = rec.tokens
    targets[idx] = rec.targets
    loss_mask[idx] = rec.loss_mask
    return tokens, targets, loss_mask


def _cfg_to_dict(cfg) -> Dict[str, Any]:
    try:
        return dataclasses.asdict(cfg)
    except Exception:
        return {f.name: getattr(cfg, f.name) for f in dc_fields(type(cfg))}


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        if o.numel() == 1:
            return o.detach().float().item() if o.is_floating_point() else o.detach().item()
        return o.detach().tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def _emit(tag: str, obj: Dict[str, Any], cfg: TrainConfig):
    obj = {**obj, "run": cfg.name}
    text = json.dumps(_jsonable(obj), default=str)
    print(f"{tag} {text}", flush=True)
    if tag == "RESULT_JSON":
        # Log collectors may split or truncate very long lines, so the result is also
        # printed as short gzip+base64 chunks: RESULT_B64 <run> <i>/<n> <chunk>
        import base64
        import gzip
        b64 = base64.b64encode(gzip.compress(text.encode("utf-8"))).decode("ascii")
        parts = [b64[i:i + 800] for i in range(0, len(b64), 800)]
        for i, part in enumerate(parts):
            print(f"RESULT_B64 {cfg.name} {i + 1}/{len(parts)} {part}", flush=True)


# ------------------------------------------------------------------ training


def run_train(cfg: TrainConfig, max_seconds=None, dry_run_steps=None) -> Dict[str, Any]:
    """Train per docs/PHASE1_DESIGN.md §4; return the RESULT_JSON dict.

    On MPS: ``torch.linalg.solve_triangular`` (used by mixer D) is not universally
    available -- run with ``PYTORCH_ENABLE_MPS_FALLBACK=1`` if the run needs it
    (documented, not worked around).
    """
    device = resolve_device(cfg.device)
    if cfg.num_threads:
        torch.set_num_threads(int(cfg.num_threads))
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % (2 ** 32))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    T = int(cfg.model.seq_len)
    model = build_model(cfg.model, device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    flops_mod = kflops
    macs_per_token = _macs_per_token(flops_mod, cfg.model, model, T)
    train_flops_tok = _train_flops_per_token(flops_mod, cfg.model, model, T)
    params_info = _param_info(model, flops_mod)

    # ---- budgets (§2: every run gets the same total training-FLOP budget F)
    if cfg.flop_budget is not None:
        flop_budget = float(cfg.flop_budget)
        token_budget = flop_budget / train_flops_tok
    else:
        flop_budget = None
        token_budget = float(cfg.token_budget)
    tokens_per_step = max(1, int(cfg.batch_size)) * T
    total_steps = max(1, int(token_budget // tokens_per_step)) if token_budget is not None else 0

    # ---- data
    val_tokens = None
    loader = None
    data_dir = _resolve_path(cfg.data_dir)
    val_path = _resolve_path(cfg.val_path)
    if cfg.task == "lm":
        if not data_dir:
            raise ValueError("task=lm needs data_dir")
        train_shards = sorted(data_dir.glob("*.bin"))[: int(cfg.n_train_shards)]
        if len(train_shards) < int(cfg.n_train_shards):
            raise ValueError(f"{data_dir}: expected >= {cfg.n_train_shards} shards, "
                             f"found {len(train_shards)}")
        loader = kdata.ShardLoader(train_shards, batch_size=cfg.batch_size, T=T,
                                   device=device, seed=cfg.seed)
        if val_path:
            val_tokens = kdata.load_shard(val_path, device)[: int(cfg.val_n_tokens)]
    recall_gen = None
    val_recall_tokens = None
    # Training docs must fit the batch width T: N is sampled from 2..min(96,
    # floor((T-6)/6)) so 6N+6 <= T (§5 samples N from 2..96; the S proxy has
    # T=1024 >= 582 so the cap only binds for short-T configs).
    # Training N per row is sampled uniformly from 2..min(128, T//6) (§5, new MQAR
    # layout: the doc + queries must fit the batch width T).
    n_max_train = max(2, min(128, T // 6))
    if cfg.task == "lm" and cfg.p_syn > 0:
        recall_gen = krecall.RecallGenerator(
            vocab_size=cfg.model.vocab_size, T=T, batch_size=cfg.batch_size,
            device=device, seed=cfg.seed + 1, n_max=n_max_train)
    if cfg.task == "recall":
        recall_gen = krecall.RecallGenerator(
            vocab_size=cfg.model.vocab_size, T=T, batch_size=cfg.batch_size,
            device=device, seed=cfg.seed + 1, n_max=n_max_train)
        # one recall row as the 1-D validation token stream (val_loss/val_features
        # score positions t -> tokens[t+1] on 1-D sequences)
        val_recall_tokens = recall_gen.next_batch().tokens[0]

    # ---- optimizer (§4: AdamW, beta=(0.9, 0.95), weight decay 0.1 on >=2-D
    #      non-embedding weights, fused on CUDA)
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay),
                            betas=(0.9, 0.95), lr=0.0, fused=(device == "cuda"))

    # ---- schedule (§4: 2% linear warm-up, cosine to final_lr_frac of peak)
    warmup_steps = max(1, int(math.ceil(cfg.warmup_frac * total_steps)))
    if total_steps <= warmup_steps:
        warmup_steps = total_steps

    def lr_for_step(step_num):
        if step_num <= warmup_steps:
            return cfg.lr * (step_num / warmup_steps if warmup_steps > 0 else 1.0)
        if total_steps <= warmup_steps:
            prog = 1.0
        else:
            prog = min(1.0, (step_num - warmup_steps) / max(1, total_steps - warmup_steps))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return cfg.lr * ((1.0 - cfg.final_lr_frac) * cos + cfg.final_lr_frac)

    # ---- evaluator for the final evaluation (§6)
    evaluator = krecall.RecallEvaluator(
        model, cfg.model, device=device, n_values=cfg.eval_recall_n_values,
        n_docs=cfg.eval_n_recall_docs, seed=cfg.seed + 2,
        chunk_rows=cfg.eval_batch_rows)

    # ---- validation points: 12 evenly spaced over the run
    val_steps = (sorted({max(1, int(round(total_steps * (k + 1) / N_VAL_POINTS)))
                         for k in range(N_VAL_POINTS)} | {total_steps})
                 if total_steps > 0 else set())

    # ---- looping (§1.3)
    loop_cfg = cfg.model.loop_end > cfg.model.loop_start
    loop_depth = int(cfg.model.n_loops) if loop_cfg else 1
    sample_min = int(cfg.model.loop_sample_min) if loop_cfg else 0

    def step_n_loops():
        if not loop_cfg:
            return None
        if sample_min > 0:
            lo = max(1, min(sample_min, loop_depth))
            return int(torch.randint(lo, loop_depth + 1, (1,)).item())
        return loop_depth

    # ---- training loop
    model.train()
    compile_model = bool(cfg.compile) and device == "cuda"
    fwd = torch.compile(model, mode=cfg.compile_mode) if compile_model else model
    amp_on = (cfg.amp == "bf16") and device in ("cuda", "cpu")
    autocast_ctx = (torch.autocast(device_type=device, dtype=torch.bfloat16)
                    if amp_on else nullcontext())
    it = iter(loader) if loader is not None else None
    pick_gen = torch.Generator().manual_seed(int(cfg.seed) * 31 + 7)

    t_start = time.perf_counter()
    log_t0, log_tok0, log_eval0 = t_start, 0, 0.0
    t_eval_total = 0.0
    t_tp0: Optional[float] = None
    tokens_tp0 = 0
    first_step_s: Optional[float] = None
    ema_loss: Optional[float] = None
    spike_count = 0
    max_grad_norm = 0.0
    diverged = False
    truncated = False
    dry_run = False
    step = 0
    tokens_done = 0
    val_curve: List[Dict[str, Any]] = []

    for step_idx in range(1, total_steps + 1):
        lr = lr_for_step(step_idx)
        for g in opt.param_groups:
            g["lr"] = lr
        n_loops = step_n_loops()
        if cfg.task == "lm":
            x, y = next(it)
            if cfg.p_syn > 0:
                bx, by, lm = _mix_recall_batch(x, y, recall_gen, cfg.p_syn, pick_gen)
            else:
                bx, by, lm = x, y, None
        else:
            rec = recall_gen.next_batch()
            bx, by, lm = rec.tokens, rec.targets, rec.loss_mask
        with autocast_ctx:
            # G3: the training step does not need logits
            out = fwd(bx, targets=by, loss_mask=lm, n_loops=n_loops,
                      return_logits=False)
        loss = out["loss"]
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip)))
        max_grad_norm = max(max_grad_norm, gn)
        opt.step()
        if step_idx == 1:
            # G4: time from process start to the end of the first optimizer step
            first_step_s = time.perf_counter() - T_PROCESS_START
        if cfg.model.jepa == "ema" and hasattr(model, "update_ema"):
            model.update_ema()
        step = step_idx
        tokens_done += int(bx.numel())
        if t_tp0 is None and (not compile_model or step > 20):
            # throughput window: skip the first 20 steps when compile is on
            t_tp0 = time.perf_counter()
            tokens_tp0 = tokens_done

        # stability: spike = logged train loss > 0.3 nats above its EMA (after warm-up)
        lv = float(loss.detach().float())
        if step > warmup_steps and ema_loss is not None and lv > ema_loss + SPIKE_NAT_ABOVE_EMA:
            spike_count += 1
        if ema_loss is None:
            ema_loss = lv
        else:
            ema_loss = SPIKE_EMA_DECAY * ema_loss + (1.0 - SPIKE_EMA_DECAY) * lv

        # logging: every 50 steps (plus step 1); tok_per_s is the rate over the
        # last logging interval, excluding evaluation time
        if step % LOG_INTERVAL == 0 or step == 1:
            aux = {k: float(out[k].detach().float()) for k in AUX_KEYS
                   if k in out and isinstance(out[k], torch.Tensor) and out[k].numel() == 1}
            now = time.perf_counter()
            rate = (tokens_done - log_tok0) / max(
                (now - log_t0) - (t_eval_total - log_eval0), 1e-9)
            _emit("LOG_JSON", {"run": cfg.name, "step": step, "loss": lv,
                               "aux": aux, "lr": lr, "grad_norm": gn,
                               "tokens": tokens_done,
                               "flops": float(tokens_done * train_flops_tok),
                               "elapsed": now - t_start,
                               "tok_per_s": rate,
                               "first_step_s": first_step_s,
                               "gpu_mem_mb": (float(torch.cuda.max_memory_allocated())
                                               / (1024 ** 2) if device == "cuda" else None)},
                  cfg)
            log_t0, log_tok0, log_eval0 = now, tokens_done, t_eval_total

        # validation at the 12 evenly spaced points
        if step in val_steps:
            t_v0 = time.perf_counter()
            vl = _validation_loss(model, cfg, val_tokens, val_recall_tokens, autocast_ctx)
            t_eval_total += time.perf_counter() - t_v0
            val_curve.append({"step": step, "tokens": tokens_done, "val_loss": vl})
            line = {"run": cfg.name, "step": step, "tokens": tokens_done, "val_loss": vl}
            if cfg.task == "recall":
                # recall metrics at the validation points too (cheap: 8 docs per N)
                # so the learning curve of exact-match is visible
                t_v1 = time.perf_counter()
                per_n = evaluator.evaluate()
                t_eval_total += time.perf_counter() - t_v1
                line["recall"] = {
                    str(N): {"exact_match_present": rm.exact_match_present,
                             "abstention_accuracy": rm.abstention_accuracy,
                             "false_abstention_rate": rm.false_abstention_rate,
                             "ece_first_answer": rm.ece_first_answer,
                             "n_queries": rm.n_queries}
                    for N, rm in per_n.items()}
            _emit("VAL_JSON", line, cfg)

        # early stop
        if dry_run_steps is not None and step >= int(dry_run_steps):
            dry_run = True
            break
        if max_seconds is not None and (time.perf_counter() - t_start) > float(max_seconds):
            truncated = True
            break

    # ---- throughput: exclude compile warm-up (first 20 steps) and all
    #      evaluation time (§4)
    t_loop_end = time.perf_counter()
    if t_tp0 is None:
        t_tp0, tokens_tp0 = t_start, 0
    elapsed = max((t_loop_end - t_tp0) - t_eval_total, 0.0)
    tok_per_s = (tokens_done - tokens_tp0) / elapsed if elapsed > 0 else 0.0

    # ---- START_JSON (§6 preflight): read budgets without training
    _emit("START_JSON", {
        "run": cfg.name,
        "params": params_info,
        "macs_per_token": macs_per_token,
        "train_flops_per_token": train_flops_tok,
        "token_budget": token_budget,
        "total_steps": total_steps,
        "device": device,
        "torch_version": torch.__version__,
        "amp": cfg.amp,
        "compile": compile_model,
        "task": cfg.task,
        "batch_size": cfg.batch_size,
        "T": T,
        "p_syn": cfg.p_syn,
    }, cfg)

    # ---- final evaluation (§6) unless diverged
    eval_out: Dict[str, Any] = {
        "val_loss_final": None, "recall": {}, "calibration": {},
        "uncertainty_signals": {}, "loop_val_losses": None, "copy_probe": None,
    }
    if not diverged and not dry_run:
        eval_out = _final_eval(model, cfg, device, val_tokens, val_recall_tokens,
                               evaluator, loop_cfg, loop_depth, autocast_ctx,
                               signal_n_tokens=cfg.signal_n_tokens)

    peak_gpu_memory_mb = None
    if device == "cuda":
        peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated()) / (1024 ** 2)

    result: Dict[str, Any] = {
        "run": cfg.name,
        "config": _cfg_to_dict(cfg),
        "device": device,
        "amp": cfg.amp,
        "compile": compile_model,
        "task": cfg.task,
        "p_syn": cfg.p_syn,
        "params": params_info,
        "macs_per_token": macs_per_token,
        "train_flops_per_token": train_flops_tok,
        "token_budget": token_budget,
        "total_flops": flop_budget if flop_budget is not None
                       else token_budget * train_flops_tok,
        "steps_run": step,
        "tokens": tokens_done,
        "elapsed_s": elapsed,
        "tok_per_s": tok_per_s,
        "first_step_s": first_step_s,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "val_curve": val_curve,
        "val_loss_vs_tokens": [{"tokens": c["tokens"], "val_loss": c["val_loss"]}
                               for c in val_curve],
        "val_loss_vs_flops": [{"flops": c["tokens"] * train_flops_tok,
                               "val_loss": c["val_loss"]} for c in val_curve],
        "bits_per_byte": None,
        "recall": {},
        "calibration": {},
        "uncertainty_signals": {},
        "loop_val_losses": None,
        "copy_probe": None,
        "stability": {"spikes": spike_count, "max_grad_norm": max_grad_norm,
                      "diverged": diverged},
        "truncated": truncated,
        "dry_run": dry_run,
    }
    result.update(eval_out)
    if result.get("val_loss_final") is not None:
        result["bits_per_byte"] = ((result["val_loss_final"] / math.log(2))
                                   * VAL_TOKENS_PER_BYTE)
    _emit("RESULT_JSON", result, cfg)
    _emit_result_b64(result, cfg)
    return result


def _final_eval(model, cfg, device, val_tokens, val_recall_tokens, evaluator,
                loop_cfg, loop_depth, autocast_ctx, signal_n_tokens=524288
                ) -> Dict[str, Any]:
    """Final evaluation producing every item of docs/PHASE1_DESIGN.md §6.

    Budget: val loss on the full 2,097,152-token validation stream; the
    uncertainty-signal table and calibration on the first `signal_n_tokens`
    (524,288) tokens; `halting_depth` and `loop_val_losses` on those same tokens.
    """
    # ---- validation loss (final): the full 2,097,152-token validation stream
    if cfg.task == "lm":
        val_loss_final = kmetrics.val_loss(model, val_tokens, cfg=cfg.model,
                                           autocast_ctx=autocast_ctx,
                                           chunk=cfg.eval_batch_rows)
    else:
        val_loss_final = kmetrics.val_loss(model, val_recall_tokens, cfg=cfg.model,
                                           autocast_ctx=autocast_ctx,
                                           chunk=cfg.eval_batch_rows)

    # ---- recall metrics (§5) + recall arrays for the §6 table
    per_n = evaluator.evaluate()
    recall_result: Dict[str, Dict[str, Any]] = {}
    recall_labels_all, recall_entropy_all, recall_correct_a1_all = [], [], []
    recall_signals_all: Dict[str, List[torch.Tensor]] = {}
    for N in sorted(per_n):
        rm = per_n[N]
        recall_result[str(N)] = {
            "exact_match_present": rm.exact_match_present,
            "abstention_accuracy": rm.abstention_accuracy,
            "false_abstention_rate": rm.false_abstention_rate,
            "ece_first_answer": rm.ece_first_answer,
            "n_queries": rm.n_queries,
        }
        recall_labels_all.append(rm.exact_match.cpu())
        recall_entropy_all.append(rm.entropy_at_key.cpu())
        recall_correct_a1_all.append(rm.correct_a1.cpu())
        for name, arr in rm.signals_at_key.items():
            recall_signals_all.setdefault(name, []).append(arr.cpu())
    recall_labels = torch.cat(recall_labels_all) if recall_labels_all else None
    recall_signals = ({k: torch.cat(v) for k, v in recall_signals_all.items()}
                      if recall_signals_all else None)
    recall_entropy_at_key = (torch.cat(recall_entropy_all) if recall_entropy_all else None)
    recall_correct_a1 = (torch.cat(recall_correct_a1_all) if recall_correct_a1_all else None)
    ece_recall_answers = None
    if (recall_signals is not None and recall_correct_a1 is not None
            and "max_prob" in recall_signals):
        ece_recall_answers = kmetrics.ece(recall_signals["max_prob"],
                                          recall_correct_a1, n_bins=15)

    # ---- calibration + signal table on the first `signal_n_tokens` (524,288)
    if cfg.task == "lm":
        # a whole number of windows plus one token, so every position gets a prediction
        T_ = int(cfg.model.seq_len)
        k_ = max(1, min(int(signal_n_tokens), val_tokens.numel() - 1) // T_)
        signal_tokens = val_tokens[: k_ * T_ + 1]
    else:
        signal_tokens = val_recall_tokens
    feat = kmetrics.val_features(model, signal_tokens, cfg=cfg.model,
                                 autocast_ctx=autocast_ctx,
                                 chunk=cfg.eval_batch_rows)

    # ---- calibration (§6): ECE + Brier on validation; entity-like restricted;
    #      ECE on recall answers
    emask = kmetrics.entity_mask(cfg.model.vocab_size,
                                  getattr(cfg, "entity_mask_path", None))
    calib: Dict[str, Any] = {
        "ece_val": kmetrics.ece(feat.conf, feat.correct, n_bins=15),
        "brier_val": kmetrics.brier(feat.conf, feat.correct),
        "ece_val_entity_like": None,
        "brier_val_entity_like": None,
    }
    if signal_tokens is not None and emask is not None and cfg.task == "lm":
        tgt = signal_tokens[1:].cpu().numpy()
        ent_m = torch.from_numpy(emask[tgt]).to(feat.conf.device)
        if bool(ent_m.any()):
            calib["ece_val_entity_like"] = kmetrics.ece(
                feat.conf[ent_m], feat.correct[ent_m], n_bins=15)
            calib["brier_val_entity_like"] = kmetrics.brier(
                feat.conf[ent_m], feat.correct[ent_m])
    calib["ece_recall_answers"] = ece_recall_answers

    # ---- uncertainty-signal table (§6), on the first signal_n_tokens
    val_signals = {"max_prob": feat.conf, "entropy": feat.entropy}
    for name in ("u_tok", "u_span", "jepa_logvar", "jepa_err", "loop_delta_last"):
        v = getattr(feat, name, None)
        if v is not None:
            val_signals[name] = v
    if loop_cfg:
        val_signals["loop_halt"] = kmetrics.halting_depth(
            model, signal_tokens, loop_depth, cfg=cfg.model,
            autocast_ctx=autocast_ctx, chunk=cfg.eval_batch_rows)
    table = kmetrics.uncertainty_table(
        val_signals, feat.correct, kmetrics.next8_labels(feat.correct), feat.entropy,
        recall_labels, recall_signals, recall_entropy_at_key)

    # ---- loop validation loss at each n ("also record validation loss at each n")
    loop_val_losses = None
    if loop_cfg:
        loop_val_losses = {str(n): kmetrics.val_loss(
            model, signal_tokens, n_loops=n, cfg=cfg.model,
            autocast_ctx=autocast_ctx, chunk=cfg.eval_batch_rows)
            for n in range(1, loop_depth + 1)}

    # ---- copy probe (lm only): how well the model copies repeated context
    copy_probe = None
    if cfg.task == "lm":
        from kenchi import copy_probe as kcopy
        copy_probe = {
            "random": kcopy.copy_probe_random(model, val_tokens, device=device,
                                              autocast_ctx=autocast_ctx,
                                              vocab_size=cfg.model.vocab_size,
                                              row_chunk=cfg.eval_batch_rows),
            "natural": kcopy.copy_probe_natural(model, val_tokens, device=device,
                                                autocast_ctx=autocast_ctx,
                                                row_chunk=cfg.eval_batch_rows),
        }

    return {
        "val_loss_final": val_loss_final,
        "recall": recall_result,
        "calibration": calib,
        "uncertainty_signals": table,
        "loop_val_losses": loop_val_losses,
        "copy_probe": copy_probe,
    }


# ------------------------------------------------------------------ CLI


def _emit_result_b64(result: Dict[str, Any], cfg: TrainConfig):
    """Emit the result as gzip+base64 chunks: `RESULT_B64 <run> <i>/<n> <chunk>`
    (800-char chunks; cluster log collectors may split long lines)."""
    blob = gzip.compress(json.dumps(_jsonable(result), default=str).encode("utf-8"))
    text = base64.b64encode(blob).decode("ascii")
    chunk = 800
    n = max(1, -(-len(text) // chunk))
    for i in range(1, n + 1):
        piece = text[(i - 1) * chunk: i * chunk]
        print(f"RESULT_B64 {cfg.name} {i}/{n} {piece}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kenchi.train")
    ap.add_argument("--config", required=True, help="config JSON path")
    ap.add_argument("--override", action="append", default=[],
                    help="key=value overrides; dotted keys like model.d_model; "
                         "values are JSON when possible")
    ap.add_argument("--max_seconds", type=float, default=None,
                    help="stop training early (still runs the final eval)")
    ap.add_argument("--dry_run_steps", type=int, default=None,
                    help="run only this many steps")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.override)
    run_train(cfg,
              max_seconds=args.max_seconds if args.max_seconds is not None else cfg.max_seconds,
              dry_run_steps=(args.dry_run_steps if args.dry_run_steps is not None
                             else cfg.dry_run_steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
