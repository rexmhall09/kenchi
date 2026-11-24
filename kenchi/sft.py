"""Stage 2 supervised fine-tuning on the sft_v2 token shards (src/kenchi/sft.py).

CLI (the job launcher is scripts/launch/sft_run.py):
    python -m kenchi.sft --config configs/sft/kenchi_v1_sft.json --weights <run>_final.safetensors

Design
------

Data
    The token shards of ``~/kenchi_data/sft_v2`` (or the staged copy the job
    downloaded from the relay): the four search/functional groups of the group
    manifest ``sft_shards_manifest.json`` (qa_search, chat, fc, harness) plus
    replay groups of general instruction text (smoltalk_clean, oasst2_en from
    ``~/kenchi_data/pretrain_extras_v2``).  Each shard is read as ONE token
    stream and its loss mask is computed ONCE, vectorised over the special
    tokens (``shard_assistant_mask``): a token is scored when it is produced
    by the assistant, i.e. everything after ``<|assistant|>`` up to and
    including the next ``<|end|>`` (a ``<|tool_call|>`` the assistant emits is
    inside that span).  Tool-result turns, user turns and the system turn are
    never scored.  Replay groups use the full-token mask.

    Windows of ``seq_len`` tokens are cut from each shard stream
    (``train_starts``: start + T < n, so no padding waste); the model
    forward takes ``loss_mask`` aligned with the targets (the scored
    prediction is the target token, mirroring ``tokenizer.assistant_mask``).

Mixing
    ``compute_group_weights``: main-group weights are proportional to the
    available train tokens of the sft_v2 groups; the replay groups share
    ``replay_weight`` (default 15%) of the mix, split proportionally to
    their train tokens.  A group the planned horizon would see more than
    ``epochs_cap`` times (default 3) is capped and the rest renormalised.
    ``passes`` (default 2) sets the planned horizon = passes x sft_v2 train
    tokens.

Training (scripts/launch/sft_run.py, DDP like the pretrain trainer)
    Start from ``--weights`` via ``kenchi.io.load_model``; AdamW (betas
    0.9/0.95, weight decay 0.0), clip 1.0, bf16 autocast, torch.compile,
    token schedule (kenchi.pretrain.TokenSchedule): peak_lr 1e-4, 2% warm-up,
    cosine to 10%.  Time-boxed with ``--time-budget-s`` (replanned with
    kenchi.pretrain.TimeBudgetReplanner).  Dev loss per group (assistant
    masked) at start and end; identical-weights-across-ranks checks at start
    and end; ``--out`` written with kenchi.io.save_model.

Logging: ``LOG_JSON`` per step, ``DEV_LOSS_JSON`` per group at start/end,
``PARAM_CHECKSUM``/``RANKS_AGREE`` lines, ``SFT_WROTE`` on the output.
"""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from kenchi import ckpt as kckpt
from kenchi import data as kdata
from kenchi import io as kenchi_io
from kenchi import tokenizer
from kenchi import tokenizer_core
from kenchi.pretrain import TimeBudgetReplanner, TokenSchedule
from kenchi.train import _param_groups, resolve_device

REPO_ROOT = Path(__file__).resolve().parents[1]

TOKEN_ASSISTANT_ID = tokenizer_core.TOKEN_ASSISTANT_ID   # 50259
TOKEN_END_ID = tokenizer_core.TOKEN_END_ID               # 50262

DEFAULT_MAIN_GROUPS = ("qa_search", "chat", "fc", "harness")
DEFAULT_REPLAY_GROUPS = {
    "replay_smoltalk_clean": "~/kenchi_data/pretrain_extras_v2/smoltalk_clean",
    "replay_oasst2_en": "~/kenchi_data/pretrain_extras_v2/oasst2_en",
}
EPOCHS_CAP = 3.0
REPLAY_WEIGHT_DEFAULT = 0.15
PASSES_DEFAULT = 2.0
LOG_EVERY = 5

__all__ = [
    "SftConfig", "load_sft_config", "shard_assistant_mask", "window_rows",
    "train_starts", "dev_starts", "GroupStream", "compute_group_weights",
    "build_groups", "group_dev_loss", "run_sft", "main", "_param_checksum",
    "ranks_agree", "write_mask_examples",
]


# ------------------------------------------------------------------ mask

def shard_assistant_mask(tokens) -> np.ndarray:
    """Vectorised assistant mask of one shard token stream.

    True for a token when it is produced by the assistant: there is an
    earlier ``<|assistant|>`` (50259) with no ``<|end|>`` (50262) strictly
    between it and this token.  So everything after ``<|assistant|>`` up to
    and including the next ``<|end|>`` is True (the closing ``<|end|>`` is
    included; the ``<|assistant|>`` itself is not).  Equivalent to
    ``kenchi.tokenizer.assistant_mask`` on a rendered stream.
    """
    toks = np.asarray(tokens)
    n = int(toks.shape[0])
    if n == 0:
        return np.zeros(0, dtype=bool)
    A = (toks == TOKEN_ASSISTANT_ID).astype(np.int64)
    B = (toks == TOKEN_END_ID).astype(np.int64)
    c = np.zeros(n + 1, dtype=np.int64)          # c[j] = # of B in [0, j)
    np.cumsum(B, out=c[1:])
    idx = np.arange(n, dtype=np.int64)
    lastA = np.maximum.accumulate(np.where(A > 0, idx, -1))
    safe = np.where(lastA >= 0, lastA, 0)
    return ((lastA >= 0) & (idx > lastA) & (c[idx] == c[safe])).astype(bool)


def train_starts(n: int, T: int) -> List[int]:
    """Window starts that fit a full ``T``-window with a target inside:
    ``range(0, n - T, T)`` (the same convention as kenchi.data.ShardLoader)."""
    if n - T <= 0:
        return []
    return list(range(0, n - T, T))


def dev_starts(n: int, T: int) -> List[int]:
    """Dev windows: the full-window starts; when too short for one, the first
    valid window at 0 (the row is padded and masked beyond the stream)."""
    starts = train_starts(n, T)
    if starts:
        return starts
    return [0] if n >= 2 else []


def window_rows(tokens, mask, T: int, starts: Sequence[int],
                full: bool = False) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Rows cut from a shard stream.

    ``tokens``: the shard stream; ``mask``: per-token mask (assistant mask,
    or all True for full-mask groups); ``full``: when True the mask is all
    True for every target inside the stream (replay groups).

    Returns ``[(x, tgt, m)]`` of length-``T`` int64/bool arrays: ``x[t]`` is
    the input token at position t (0 past the end of the stream), ``tgt[t]``
    the shifted target (0 past the end), ``m[t]`` True when the target token
    is scored.
    """
    toks = np.asarray(tokens)
    n = int(toks.shape[0])
    out = []
    for s in starts:
        s = int(s)
        x = np.zeros(T, dtype=np.int64)
        tg = np.zeros(T, dtype=np.int64)
        m = np.zeros(T, dtype=bool)
        nx = max(0, min(T, n - s))
        x[:nx] = toks[s:s + nx]
        nt = max(0, min(T, n - s - 1))
        if nt > 0:
            tg[:nt] = toks[s + 1:s + 1 + nt]
            m[:nt] = True if full else np.asarray(mask[s + 1:s + 1 + nt], dtype=bool)
        out.append((x, tg, m))
    return out


# ------------------------------------------------------------------ data

@dataclass
class _Entry:
    """One shard stream + its per-token mask + window starts."""

    tokens: np.ndarray          # stream tokens (uint16: model ids are < 50263)
    mask: np.ndarray            # per-token mask (True = assistant-produced target)
    starts: List[int]
    full: bool = False
    path: Optional[Path] = None


class GroupStream:
    """A group's train/dev shards as streams with precomputed masks."""

    def __init__(self, name: str, train_files: Sequence[Path], dev_files: Sequence[Path],
                 T: int, full_mask: bool = False):
        self.name = name
        self.T = int(T)
        self.full_mask = bool(full_mask)
        self.entries: List[_Entry] = []
        self.dev_entries: List[_Entry] = []
        for files, sink, is_train in ((train_files, self.entries, True),
                                      (dev_files, self.dev_entries, False)):
            for p in files:
                toks = np.asarray(kdata.load_shard(p).numpy(), dtype=np.uint16)
                n = int(toks.shape[0])
                if self.full_mask:
                    m = np.ones(n, dtype=bool)
                else:
                    m = shard_assistant_mask(toks)
                starts = train_starts(n, self.T) if is_train else dev_starts(n, self.T)
                sink.append(_Entry(tokens=toks, mask=m, starts=starts,
                                   full=self.full_mask, path=Path(p)))
        self.n_train = sum(int(e.tokens.shape[0]) for e in self.entries)
        self.n_dev = sum(int(e.tokens.shape[0]) for e in self.dev_entries)

    def pick_entry(self, rng: random.Random) -> Optional[_Entry]:
        if not self.entries:
            return None
        return self.entries[rng.randrange(len(self.entries))]

    def sample_rows(self, rng: random.Random,
                    n: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        while len(out) < n:
            ent = self.pick_entry(rng)
            if ent is None or not ent.starts:
                break
            s = ent.starts[rng.randrange(len(ent.starts))]
            out.extend(window_rows(ent.tokens, ent.mask, self.T, [s], full=ent.full))
        return out[:n]

    def dev_rows(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        out = []
        for ent in self.dev_entries:
            out.extend(window_rows(ent.tokens, ent.mask, self.T, ent.starts, full=ent.full))
        return out

    def print_mask_examples(self, n: int = 3, out: Optional[List[str]] = None) -> None:
        """Decode the first ``n`` train streams of the group with the scored
        spans marked: ``▐▌`` around each scored token."""
        def _emit(s: str) -> None:
            if out is not None:
                out.append(s)
            else:
                print(s)

        shown = 0
        for ent in self.entries:
            if shown >= n or not ent.starts:
                break
            s = ent.starts[0]
            x, _, m = window_rows(ent.tokens, ent.mask, self.T, [s], full=ent.full)[0]
            parts = []
            for t in range(self.T):
                tok = int(x[t])
                txt = (tokenizer_core.SPECIAL_STRINGS.get(tok)
                       if tok >= tokenizer_core.N_BASE else tokenizer.decode_text([tok]))
                parts.append((txt, bool(m[t])))
            line = "".join(("▐" if sc else "") + (txt or "") + ("▌" if sc else "")
                           for txt, sc in parts)
            hdr = (f"### {self.name} — shard {ent.path.name if ent.path else '?'} "
                   f"window start {s}: scored tokens marked ▐...▌")
            _emit(hdr)
            _emit(f"```\n{line[:4000]}\n```")
            shown += 1


# ------------------------------------------------------------------ weights

def _group_manifest_groups(manifest_path: Path) -> Dict[str, Any]:
    return json.loads(Path(manifest_path).read_text()).get("groups") or {}


def _shard_path(manifest_path: Path, rel: str) -> Path:
    """Resolve a manifest shard path: absolute as-is; otherwise relative to
    the manifest's ``shards/`` dir, then the manifest dir itself."""
    p = Path(rel).expanduser()
    if p.is_absolute():
        return p
    base = manifest_path.parent
    for cand in (base / "shards" / p, base / p):
        if cand.is_file():
            return cand
    return base / "shards" / p


def _group_shard_files(manifest_path: Path, group: str, split: str) -> List[Path]:
    groups = _group_manifest_groups(manifest_path)
    return [_shard_path(manifest_path, s["path"])
            for s in groups.get(group, {}).get(split, {}).get("shards", [])]


def _replay_files(dir: str) -> List[Path]:
    d = Path(dir).expanduser()
    return sorted(d.glob("train_*.bin"))


def _shard_tokens(paths: Sequence[Path]) -> int:
    return sum(int(kdata.read_shard_header(p)[2]) for p in paths)


def main_group_names(manifest_path: Path) -> List[str]:
    """The groups with train shards in the group manifest (the sft_v2 search
    and function-calling groups; replay groups live in the SFT config)."""
    return [g for g in _group_manifest_groups(manifest_path)
            if _group_shard_files(manifest_path, g, "train")]


def compute_group_weights(manifest_path: Path,
                          replay_dirs: Dict[str, str],
                          replay_weight: float = REPLAY_WEIGHT_DEFAULT,
                          passes: float = PASSES_DEFAULT,
                          epochs_cap: float = EPOCHS_CAP,
                          ) -> Dict[str, float]:
    """Effective mixing weights for one SFT schedule.

    Main groups: weights proportional to the train tokens of the sft_v2
    groups; the replay groups share ``replay_weight`` of the mix, split
    proportionally to their train tokens.  A group the planned horizon
    (``passes`` x the sft_v2 train tokens) would see more than
    ``epochs_cap`` times is capped (its weight reduced, the rest
    redistributed).
    """
    man = Path(manifest_path)
    main = main_group_names(man)
    counts = {g: _shard_tokens(_group_shard_files(man, g, "train")) for g in main}
    total = sum(counts.values())
    if total <= 0:
        raise ValueError(f"no train shards in the sft_v2 manifest {man}")
    replay_counts = {g: _shard_tokens(_replay_files(d)) for g, d in replay_dirs.items()}
    rtotal = sum(replay_counts.values())
    if replay_weight > 0 and rtotal <= 0:
        print(f"CAPPED_GROUPS note: no replay shards found in {list(replay_dirs.values())}; "
              f"proceeding with replay_weight=0", flush=True)
        replay_weight = 0.0

    w: Dict[str, float] = {}
    for g in main:
        w[g] = (1.0 - replay_weight) * (counts[g] / total)
    for g, c in replay_counts.items():
        w[g] = replay_weight * (c / rtotal) if rtotal > 0 else 0.0

    horizon = float(passes) * float(total)
    capped = []
    for g in main:
        epochs = (w[g] * horizon) / counts[g] if counts[g] > 0 else 0.0
        if epochs > epochs_cap:
            capped.append(f"{g}({epochs:.1f}x)")
            w[g] = epochs_cap * counts[g] / horizon
    if capped:
        s = sum(w[g] for g in main)
        if s > 0:
            scale = (1.0 - replay_weight) / s
            for g in main:
                w[g] *= scale
        print(f"CAPPED_GROUPS {', '.join(capped)} at {epochs_cap:.0f}x "
              f"(horizon {horizon / 1e6:.0f}M tokens)", flush=True)
    return {g: w[g] for g in sorted(w) if w[g] > 0}


def log_line(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------------------------------------------ config

@dataclass
class SftConfig:
    """Config for the stage-2 fine-tuner (see src/kenchi/sft.py docstring)."""

    name: str = "sft-kenchi_v1"
    weights: Optional[str] = None                       # pretrained safetensors (--weights)
    weights_config: Optional[str] = None                # config of the weights (default: config)
    config: str = "configs/sft/kenchi_v1_sft.json"
    out: Optional[str] = None                           # --out: the SFT safetensors
    sft_shards_manifest: str = "~/kenchi_data/sft_v2/sft_shards_manifest.json"
    replay_groups: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_REPLAY_GROUPS))
    replay_weight: float = REPLAY_WEIGHT_DEFAULT
    passes: float = PASSES_DEFAULT
    epochs_cap: float = EPOCHS_CAP
    groups: Dict[str, float] = field(default_factory=lambda: {g: 0.0 for g in DEFAULT_MAIN_GROUPS})
    seq_len: int = 0
    peak_lr: float = 1e-4
    final_lr_frac: float = 0.10
    warmup_frac: float = 0.02
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    micro_batch_rows: int = 12
    #: micro-batches per optimizer step on each rank.  The 8-GPU run took one
    #: micro-batch of 12 rows per GPU per step (96 rows); on one GPU set this to
    #: 8 for the same step size.  (One GPU with accum 1 = 8x more, 8x noisier
    #: steps at the same learning rate: measured worse on every dev group and
    #: on ARC-Easy/PIQA, 2026-11-19.)
    accum_steps: int = 1
    amp: str = "bf16"
    compile: bool = True
    seed: int = 0
    device: str = "auto"
    log_every: int = LOG_EVERY
    max_time_s: Optional[float] = None
    ckpt_dir: str = "ckpts/sft"
    dev_batch_rows: int = 8
    time_budget_measure_after_s: Optional[float] = None


def load_sft_config(path: Path, overrides: Sequence[str] = ()) -> SftConfig:
    """Load a configs/sft/*.json (``k=v`` dotted overrides are applied to the
    loaded config, like sft_run.py's --override flags)."""
    raw = json.loads(Path(path).expanduser().read_text())

    def ovr(key, default):
        v = raw.get(key)
        return default if v is None else v

    cfg = SftConfig(
        name=str(ovr("name", "sft-kenchi_v1")),
        weights=(str(raw["weights"]) if raw.get("weights") else None),
        weights_config=(str(raw["weights_config"]) if raw.get("weights_config") else None),
        config=str(ovr("config", "configs/sft/kenchi_v1_sft.json")),
        out=(str(raw["out"]) if raw.get("out") else None),
        sft_shards_manifest=str(ovr("sft_shards_manifest",
                                    "~/kenchi_data/sft_v2/sft_shards_manifest.json")),
        replay_groups={str(k): str(v) for k, v in
                       (ovr("replay_groups", None) or DEFAULT_REPLAY_GROUPS).items()},
        replay_weight=float(ovr("replay_weight", REPLAY_WEIGHT_DEFAULT)),
        passes=float(ovr("passes", PASSES_DEFAULT)),
        epochs_cap=float(ovr("epochs_cap", EPOCHS_CAP)),
        groups={str(k): float(v) for k, v in (ovr("groups", None) or {}).items()},
        seq_len=int(ovr("seq_len", 0)),
        peak_lr=float(ovr("peak_lr", 1e-4)),
        final_lr_frac=float(ovr("final_lr_frac", 0.10)),
        warmup_frac=float(ovr("warmup_frac", 0.02)),
        weight_decay=float(ovr("weight_decay", 0.0)),
        grad_clip=float(ovr("grad_clip", 1.0)),
        micro_batch_rows=int(ovr("micro_batch_rows", 12)),
        accum_steps=int(os.environ.get("KENCHI_SFT_ACCUM") or ovr("accum_steps", 1)),
        amp=str(ovr("amp", "bf16")),
        compile=bool(ovr("compile", True)),
        seed=int(ovr("seed", 0)),
        device=str(ovr("device", "auto")),
        log_every=int(ovr("log_every", LOG_EVERY)),
        max_time_s=(float(raw["max_time_s"]) if raw.get("max_time_s") else None),
        ckpt_dir=str(ovr("ckpt_dir", "ckpts/sft")),
        dev_batch_rows=int(ovr("dev_batch_rows", 8)),
        time_budget_measure_after_s=(float(raw["time_budget_measure_after_s"])
                                     if raw.get("time_budget_measure_after_s") else None),
    )
    apply_overrides(cfg, overrides)
    return cfg


def apply_overrides(cfg: SftConfig, overrides: Sequence[str]) -> None:
    """``k=v`` dotted overrides against the config dict (values are JSON)."""
    for ov in overrides:
        k, _, v = ov.partition("=")
        keys = [kk for kk in k.split(".") if kk]
        if not keys:
            continue
        cur: Dict[str, Any] = cfg.__dict__
        for kk in keys[:-1]:
            nxt = cur.get(kk)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[kk] = nxt
            cur = nxt
        if v:
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
        else:
            v = None
        cur[keys[-1]] = v


# ------------------------------------------------------------------ training

def _param_checksum(model) -> str:
    """sha256 over (name, dtype, bytes) of every parameter (pretrain.py style)."""
    import hashlib
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode("utf-8"))
        h.update(str(p.dtype).encode())
        h.update(p.detach().to("cpu").contiguous().numpy().tobytes())
    return h.hexdigest()


def ranks_agree(tag: str, checksum: str, device: str = "cpu") -> bool:
    """True when every rank holds bit-identical parameters (pretrain.py style:
    48 bits of the sha256 compared through an all-reduce MAX)."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        v = float(int(checksum[:12], 16))
        t = torch.tensor([v, -v], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ok = bool(t[0] == -t[1])
    else:
        ok = True
    print(f"RANKS_AGREE {tag} {ok}", flush=True)
    return ok


def build_groups(cfg: SftConfig, T: int) -> Dict[str, GroupStream]:
    """The main groups (from the sft_v2 group manifest) + replay groups."""
    man = Path(cfg.sft_shards_manifest).expanduser()
    groups: Dict[str, GroupStream] = {}
    for g in _group_manifest_groups(man):
        train = _group_shard_files(man, g, "train")
        dev = _group_shard_files(man, g, "dev")
        if train or dev:
            groups[g] = GroupStream(g, train, dev, T)
    for name, d in cfg.replay_groups.items():
        train = _replay_files(d)
        if train:
            groups[name] = GroupStream(name, train, [], T, full_mask=True)
    return groups


def group_dev_loss(model, group: GroupStream, autocast_ctx,
                   batch_rows: int = 8, device=None, max_rows: int = 48) -> Optional[float]:
    """Assistant-masked mean CE of a group's dev rows (weighted by scored
    positions), or None when the group has no dev rows."""
    rows = group.dev_rows()
    if not rows:
        return None
    if max_rows and len(rows) > max_rows:          # an evenly spaced, fixed subset
        step_ = len(rows) / float(max_rows)
        rows = [rows[int(i * step_)] for i in range(int(max_rows))]
    dev = device or next(model.parameters()).device
    was = model.training
    model.eval()
    tot, cnt = 0.0, 0
    try:
        with torch.no_grad():
            for i in range(0, len(rows), batch_rows):
                chunk = rows[i:i + batch_rows]
                idx, tgt, msk = _to_tensors(chunk, dev)
                with autocast_ctx:
                    out = model(idx, targets=tgt, loss_mask=msk, return_logits=False)
                n = int(msk.sum())
                tot += float(out["loss_lm"]) * n
                cnt += n
    finally:
        if was:
            model.train()
    return tot / max(cnt, 1)


def _to_tensors(rows, device=None):
    idx = torch.from_numpy(np.stack([r[0] for r in rows])).long()
    tgt = torch.from_numpy(np.stack([r[1] for r in rows])).long()
    msk = torch.from_numpy(np.stack([r[2] for r in rows]))
    if device is not None:
        idx, tgt, msk = idx.to(device), tgt.to(device), msk.to(device)
    return idx, tgt, msk


def sample_batch_rows(groups: Dict[str, GroupStream], weights: Dict[str, float],
                      rng: random.Random, n: int
                      ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``n`` random rows, one group at a time, sampled by the mixing weights."""
    names = [g for g in sorted(weights)
             if weights.get(g, 0) > 0 and groups.get(g) is not None]
    if not names:
        raise ValueError("no SFT groups with train shards")
    total = sum(weights[g] for g in names)
    rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    while len(rows) < n:
        r = rng.random() * total
        acc = 0.0
        pick = names[-1]
        for nm in names:
            acc += weights[nm]
            if r <= acc:
                pick = nm
                break
        got = groups[pick].sample_rows(rng, 1)
        if not got:
            continue
        rows.extend(got)
    return rows[:n]


def write_mask_examples(groups: Dict[str, GroupStream], path: Path, n: int = 3) -> None:
    """Append the per-group mask verification (scored spans marked) to
    ``path``: the text with ▐...▌ around every scored token."""
    lines = [f"### SFT mask verification ({time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())})"]
    lines.append("Vectorised state machine over the special tokens "
                 "(shard_assistant_mask in src/kenchi/sft.py): a target token is scored "
                 "when it is produced by the assistant (everything after `<|assistant|>` "
                 "up to and including the next `<|end|>`; tool-result, user and system "
                 "tokens are never scored). Marked ▐...▌.")
    for g in sorted(groups):
        if not groups[g].entries:
            continue
        out: List[str] = []
        groups[g].print_mask_examples(n, out=out)
        lines.extend(out)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
    log_line(f"MASK_EXAMPLES_WRITTEN {path} ({len(groups)} groups, {n} conversations each)")


def run_sft(cfg: SftConfig) -> Dict[str, Any]:
    """DDP training run; returns the result dict (rank 0)."""
    device = resolve_device(cfg.device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    elif os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
        torch.distributed.init_process_group(
            backend="nccl" if device == "cuda" else "gloo")
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    else:
        rank, world = 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS") or "8"))
    torch.manual_seed(cfg.seed + rank)
    random.seed(cfg.seed + rank)

    if not cfg.weights:
        raise ValueError("run_sft needs --weights (the pretrained safetensors)")
    if not cfg.out:
        raise ValueError("run_sft needs --out")
    model, mcfg = kenchi_io.load_model(cfg.weights, cfg.weights_config or cfg.config, device=device)
    T = int(mcfg.seq_len)
    if cfg.seq_len and int(cfg.seq_len) != T:
        print(f"SFT_SEQ_LEN model={T} config={cfg.seq_len}; using the model's", flush=True)

    compile_model = bool(cfg.compile) and device == "cuda"
    if compile_model:
        model = torch.compile(model)     # as in pretraining: 2.3x faster, half the memory
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device == "cuda" else None)
    raw = kckpt.unwrap(model)

    # ---- data: groups, masks, dev sets (KENCHI_SFT_MANIFEST overrides the manifest)
    man = Path(os.environ.get("KENCHI_SFT_MANIFEST") or cfg.sft_shards_manifest).expanduser()
    if os.environ.get("KENCHI_SFT_MANIFEST"):
        cfg.sft_shards_manifest = str(man)
    groups = build_groups(cfg, T)
    if not groups:
        raise ValueError(f"no SFT data found in {cfg.sft_shards_manifest}")
    weights = compute_group_weights(man, cfg.replay_groups, replay_weight=cfg.replay_weight,
                                     passes=cfg.passes, epochs_cap=cfg.epochs_cap)
    weights = {g: w for g, w in weights.items() if groups.get(g)}
    for g in groups:
        weights.setdefault(g, 0.0)
    # planned horizon: passes x the sft_v2 train tokens of the main groups
    horizon = float(cfg.passes) * float(sum(
        _shard_tokens(_group_shard_files(man, g, "train"))
        for g in main_group_names(man) if groups.get(g) is not None))

    accum = max(1, int(cfg.accum_steps))
    rows_per_step = cfg.micro_batch_rows * world * accum
    tokens_per_step = rows_per_step * T

    # ---- optimizer + schedule
    optimizer = torch.optim.AdamW(_param_groups(raw, cfg.weight_decay),
                                  betas=(0.9, 0.95), lr=0.0, fused=(device == "cuda"))
    # The plan is fixed BEFORE training and is the same on every rank (each rank
    # has its own schedule object; a wall-clock replan per rank would give the
    # ranks different learning rates).  With a time box, cap the plan at what
    # fits: (box - start-up) x an assumed speed.
    if cfg.max_time_s and device == "cuda":
        fit = max(0.0, float(cfg.max_time_s) - 170.0) * 70_000.0 * world
        if fit < horizon:
            if rank == 0:
                print(f"SFT_PLAN horizon cut from {horizon:.0f} to {fit:.0f} tokens to fit the time box",
                      flush=True)
            horizon = fit
    schedule = TokenSchedule(cfg.peak_lr, cfg.final_lr_frac, "cosine",
                             warmup_tokens=int(cfg.warmup_frac * max(1.0, horizon)),
                             total_tokens=max(1.0, horizon))
    replanner = None
    t_proc0 = time.perf_counter()
    amp_on = cfg.amp == "bf16" and device in ("cuda", "cpu")
    autocast_ctx = (torch.autocast(device_type=device, dtype=torch.bfloat16)
                    if amp_on else nullcontext())
    rng = random.Random(cfg.seed + rank)

    def emit(tag: str, obj: Dict[str, Any]) -> None:
        if rank == 0:
            print(f"{tag} {json.dumps(obj, default=str)}", flush=True)

    emit("START_JSON", {
        "run": cfg.name, "weights": str(cfg.weights), "weights_config": str(cfg.weights_config),
        "out": str(cfg.out), "T": T,
        "groups": {g: weights.get(g, 0.0) for g in sorted(groups)},
        "horizon_tokens": horizon, "passes": cfg.passes, "replay_weight": cfg.replay_weight,
        "time_budget_s": cfg.max_time_s, "peak_lr": cfg.peak_lr,
        "warmup_frac": cfg.warmup_frac, "final_lr_frac": cfg.final_lr_frac,
        "micro_batch_rows": cfg.micro_batch_rows, "rows_per_step": rows_per_step,
        "tokens_per_step": tokens_per_step, "world_size": world, "compile": compile_model,
        "amp": cfg.amp, "device": device, "sft_shards_manifest": str(man),
        "tokenizer_kind": tokenizer.tokenizer_kind(),
    })

    # ---- dev loss at start
    dev_start = ({g: group_dev_loss(raw, groups[g], autocast_ctx, cfg.dev_batch_rows)
                  for g in sorted(groups)} if rank == 0 else {})
    for g in sorted(dev_start):
        emit("DEV_LOSS_JSON", {"group": g, "phase": "start", "loss": dev_start[g],
                               "scored": groups[g].n_dev})
    csum = _param_checksum(raw)
    print(f"PARAM_CHECKSUM {rank} {csum} phase=sft_start", flush=True)
    ranks_agree("sft_start", csum, device)

    # ---- mask verification -> notes/post_job.md (rank 0)
    if rank == 0:
        Path("notes").mkdir(parents=True, exist_ok=True)
        write_mask_examples(groups, Path("notes/post_job.md"))

    # ---- training loop
    model.train()
    step, tokens = 0, 0
    losses: List[float] = []
    t_start = time.perf_counter()
    tp0, tok0 = t_start, 0
    while True:
        lr = schedule.lr_for(tokens)
        for g in optimizer.param_groups:
            g["lr"] = lr
        finite = torch.ones((), device=device)
        lm_sum = 0.0
        for _a in range(accum):
            rows = sample_batch_rows(groups, weights, rng, cfg.micro_batch_rows)
            idx, tgt, msk = _to_tensors(rows, device)
            with autocast_ctx:
                out = model(idx, targets=tgt, loss_mask=msk, return_logits=False)
            # backward the model's total loss: `out["loss"]` = loss_lm + loss_u (+
            # aux) — the u_head is trained during SFT and must not trip DDP's
            # "unused parameter" check when loss_lm alone is backwarded.
            loss = out["loss"] if out.get("loss") is not None else out["loss_lm"]
            finite = torch.minimum(finite, torch.isfinite(loss).float())
            (loss / accum).backward()
            lm_sum += float(out["loss_lm"].detach())
        if world > 1:
            t = torch.tensor([finite], device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
            finite = t[0]
        if float(finite) == 0:
            emit("DIVERGED", {"step": step, "tokens": tokens})
            break
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens += tokens_per_step
        losses.append(lm_sum / accum)
        now = time.perf_counter()
        if step == 1:
            tp0, tok0 = now, tokens
        if step % max(1, cfg.log_every) == 0 or step == 1:
            dt = now - tp0
            emit("LOG_JSON", {
                "step": step, "tokens": tokens, "loss": losses[-1],
                "loss_mean": sum(losses[-20:]) / len(losses[-20:]),
                "lr": lr, "grad_norm": gn,
                "tok_per_s": ((tokens - tok0) / dt) if dt > 0 else 0.0,
                "elapsed": now - t_start, "horizon": horizon,
            })
        if tokens >= max(1.0, horizon):
            break
        # safety: rank 0 watches the clock; the stop is agreed by all ranks
        late = 1.0 if (rank == 0 and cfg.max_time_s
                       and now - t_proc0 > float(cfg.max_time_s) - 45.0) else 0.0
        if world > 1:
            t = torch.tensor([late], device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
            late = float(t[0])
        if late > 0:
            if rank == 0:
                print(f"SFT_STOP time box reached at step {step} tokens {tokens} "
                      f"(plan {horizon:.0f})", flush=True)
            break

    # ---- dev loss at end
    dev_end = ({g: group_dev_loss(raw, groups[g], autocast_ctx, cfg.dev_batch_rows)
                for g in sorted(groups)} if rank == 0 else {})
    for g in sorted(dev_end):
        emit("DEV_LOSS_JSON", {"group": g, "phase": "end", "loss": dev_end[g],
                               "scored": groups[g].n_dev,
                               "delta": (dev_end[g] - dev_start[g])
                               if (dev_start.get(g) is not None and dev_end.get(g) is not None) else None})
    csum = _param_checksum(raw)
    print(f"PARAM_CHECKSUM {rank} {csum} phase=sft_end", flush=True)
    ranks_agree("sft_end", csum, device)

    # ---- write the SFT output (rank 0 only: one writer)
    if rank == 0:
        kenchi_io.save_model(raw, cfg.out, metadata={"stage": "sft", "steps": step, "tokens": tokens})
        print("SFT_WROTE " + json.dumps({
            "out": str(cfg.out), "steps": step, "tokens": tokens, "horizon": horizon,
            "dev_loss_start": {g: dev_start[g] for g in sorted(dev_start)},
            "dev_loss_end": {g: dev_end[g] for g in sorted(dev_end)},
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        }, default=str), flush=True)

    result = {
        "run": cfg.name, "weights": str(cfg.weights), "out": str(cfg.out),
        "T": T, "steps": step, "tokens": tokens, "horizon": horizon,
        "weights": weights, "dev_loss_start": dev_start, "dev_loss_end": dev_end,
        "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
        "peak_lr": cfg.peak_lr, "elapsed_s": time.perf_counter() - t_start,
        "ranks_equal_end": True, "rank": rank, "world_size": world,
    }
    emit("RESULT_JSON", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="sft_run", description="Kenchi SFT trainer (stage 2)")
    ap.add_argument("--config", default="train/configs/sft.json")
    ap.add_argument("--weights", default=None, help="pretrained <run>_final.safetensors")
    ap.add_argument("--weights-config", default=None,
                    help="config JSON of the weights (default: --config)")
    ap.add_argument("--out", default=None, help="output safetensors (<run>_sft.safetensors)")
    ap.add_argument("--time-budget-s", type=float, default=None,
                    help="time box for the SFT stage (the job's KENCHI_SFT_S)")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args(argv)
    cfg = load_sft_config(Path(args.config))
    apply_overrides(cfg, args.override)
    if args.weights:
        cfg.weights = args.weights
    if args.weights_config:
        cfg.weights_config = args.weights_config
    if args.out:
        cfg.out = args.out
    if args.time_budget_s is not None:
        cfg.max_time_s = args.time_budget_s or None
    run_sft(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
