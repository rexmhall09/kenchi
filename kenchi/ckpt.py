"""Checkpoints for the Phase 2 trainer (src/kenchi/ckpt.py).

New scheme (see notes/pretrain_trainer.md):

- **periodic resume** checkpoint every ``ckpt_every_s`` (default 1500):
  model weights **and** AdamW state, both stored as ``bfloat16``, plus
  ``step``/``tokens``/schedule/stream/RNG/config.  Rank 0 snapshots to CPU
  clones at a step boundary; serialisation + upload happen on ONE background
  thread; if the previous upload is still running the new checkpoint is
  skipped and logged (uploads are never queued).  The last 2 are kept locally;
  the relay also keeps its last 2.
- **final** artifacts (written synchronously, with retries until the relay
  confirms via its sha256+size verification):
  (a) ``<run>_final.safetensors`` -- fp32 weights (torch.save fallback when the
  ``safetensors`` package is missing) plus ``<run>_final_config.json``;
  (b) ``resume_<run>_final.pt`` -- the final resume checkpoint (fp32).
- Uploads go through ``relay_client`` (stdlib); failures are logged and never
  crash the trainer (except the final uploads, which are retried until
  ``final_upload_max_s``).

``Checkpointer`` gates saving to rank 0; every rank may still ``restore``.
On SIGTERM the handler only sets a flag: the trainer notices it through the
rank-0 control vector (see ``src/kenchi/pretrain.py`` A) and rank 0 does a
best-effort synchronous save of a periodic resume checkpoint.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

log = logging.getLogger("kenchi.ckpt")

__all__ = ["Checkpointer", "save_full", "load_full", "load_light",
           "init_from", "ensure_local", "rng_snapshot", "rng_restore", "upload_async",
           "checkpoints_on_disk", "unwrap", "FULL_FORMAT", "RESUME_FORMAT", "LIGHT_FORMAT"]

FULL_FORMAT = "kenchi-full-ckpt-v1"       # final resume ckpt: fp32 weights + optimizer + state
RESUME_FORMAT = "kenchi-resume-ckpt-v1"   # periodic resume ckpt: bf16 weights + optimizer + state
LIGHT_FORMAT = "kenchi-light-ckpt-v1"     # weights-only, bf16 (kept for compatibility)


# ------------------------------------------------------------------ helpers

def unwrap(model):
    """The underlying ``Kenchi`` module under DDP and/or torch.compile wrappers."""
    while True:
        m = getattr(model, "module", None)
        if isinstance(m, torch.nn.Module):
            model = m
            continue
        m = getattr(model, "_orig_mod", None)
        if isinstance(m, torch.nn.Module):
            model = m
            continue
        return model


def _clones_cpu(module, dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in module.state_dict().items():
        t = v.detach()
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)          # cast on the GPU: half the bytes to copy
        t = t.to("cpu")
        if t.data_ptr() == v.data_ptr():
            t = t.clone()
        out[k] = t
    return out


def _clone_opt_state(optimizer, dtype: Optional[torch.dtype] = None) -> Dict[str, Any]:
    """A CPU clone of an optimizer state dict; with ``dtype`` set, floating tensors
    (except the AdamW ``step`` counter) are cast to it."""
    sd = optimizer.state_dict()
    out = {"state": {}, "param_groups": copy.deepcopy(sd.get("param_groups", []))}
    for k, v in sd.get("state", {}).items():
        entry: Dict[str, Any] = {}
        for kk, vv in v.items():
            if isinstance(vv, torch.Tensor):
                t = vv.detach()
                if dtype is not None and kk != "step" and t.is_floating_point():
                    t = t.to(dtype)
                t = t.to("cpu")
                if t.data_ptr() == vv.data_ptr():
                    t = t.clone()
                entry[kk] = t
            else:
                entry[kk] = copy.deepcopy(vv)
        out["state"][int(k)] = entry
    return out


def with_suppress(fn, *a):
    try:
        fn(*a)
    except Exception:  # noqa: BLE001
        pass


def rng_snapshot() -> Dict[str, Any]:
    d = {
        "torch": torch.get_rng_state().clone(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            d["torch_cuda"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
        except Exception:  # noqa: BLE001
            d["torch_cuda"] = None
    return d


def rng_restore(d: Dict[str, Any]) -> None:
    if d.get("torch") is not None:
        torch.set_rng_state(d["torch"].to("cpu").to(torch.uint8))
    if d.get("numpy") is not None:
        np.random.set_state(d["numpy"])
    if d.get("python") is not None:
        random.setstate(d["python"])
    if d.get("torch_cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.to("cpu") for s in d["torch_cuda"]])
        except Exception:  # noqa: BLE001
            pass


def _state_of(obj: Any) -> Any:
    return obj.state() if hasattr(obj, "state") and callable(obj.state) else obj


def _relay():
    root = Path(__file__).resolve().parents[1]
    relay_dir = root / "train"
    if relay_dir.is_dir() and str(relay_dir) not in os.environ.get("PYTHONPATH", ""):
        import sys
        if str(relay_dir) not in sys.path:
            sys.path.insert(0, str(relay_dir))
    import relay_client
    return relay_client


def upload_async(local: Union[str, Path], remote: Optional[str] = None,
                 base: Optional[str] = None, token: Optional[str] = None):
    """Upload ``local`` to the relay in a background thread; errors are logged only."""
    rc = _relay()
    return rc.upload_async(str(local), remote or Path(local).name, base=base, token=token)


def _step_of(name: str) -> float:
    if "_final" in name:
        return float("inf")
    m = re.search(r"_(\d{12})\.pt$", name)
    return int(m.group(1)) if m else -1


def checkpoints_on_disk(ckpt_dir: Union[str, Path], kind: str,
                        run_name: Optional[str] = None) -> List[Path]:
    """All ``<kind>_<run>_*.pt`` files in ``ckpt_dir``, oldest first."""
    d = Path(ckpt_dir)
    if not d.is_dir():
        return []
    pat = f"{kind}_*.pt" if run_name is None else f"{kind}_{run_name}_*.pt"
    return sorted((p for p in d.glob(pat)), key=lambda p: _step_of(p.name))


# ------------------------------------------------------------------ save/load

def save_full(path: Union[str, Path], model, optimizer=None, *,
              step: int = 0, tokens: int = 0, schedule=None, stream=None,
              config=None, rng: Optional[Dict[str, Any]] = None,
              meta: Optional[Dict[str, Any]] = None, wait: bool = True,
              tokenizer_sha256: Optional[str] = None) -> Path:
    """Write the final resume checkpoint: fp32 weights + optimizer state +
    step/tokens/schedule/stream/RNG state + config (+ tokenizer sha).
    ``wait=False`` writes on a daemon thread after a synchronous CPU snapshot."""
    path = Path(path)
    snap = {
        "format": FULL_FORMAT,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": int(step),
        "tokens": int(tokens),
        "tokenizer_sha256": tokenizer_sha256,
        "model": _clones_cpu(unwrap(model), torch.float32),
        "optimizer": _clone_opt_state(optimizer) if optimizer is not None else None,
        "schedule": _state_of(schedule),
        "stream": _state_of(stream),
        "rng": rng if rng is not None else rng_snapshot(),
        "config": config or {},
        "meta": meta or {},
    }

    def _do():
        _write(path, snap)

    if wait:
        _do()
    else:
        threading.Thread(target=_do, name="kenchi-ckpt-save", daemon=True).start()
    return path


def _write(path: Path, snap: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save(snap, tmp)
    os.replace(tmp, path)
    log.info("[ckpt] wrote %s (%d bytes)", path, path.stat().st_size)


def load_full(path: Union[str, Path], model=None, optimizer=None) -> Dict[str, Any]:
    """Load a resume/final checkpoint; applies model + optimizer state when given.

    Returns the raw dict (``step``, ``tokens``, ``schedule``, ``stream``, ``rng``,
    ``config``, ``meta``) so the caller can restore the rest.
    """
    d = torch.load(Path(path), map_location="cpu", weights_only=False)
    if model is not None and d.get("model") is not None:
        unwrap(model).load_state_dict(d["model"], strict=True)
    if optimizer is not None and d.get("optimizer") is not None:
        optimizer.load_state_dict(d["optimizer"])
    return d


def init_from(path_or_name: Union[str, Path], model, base: Optional[str] = None,
              token: Optional[str] = None) -> Dict[str, Any]:
    """Load weights only (``--init_from``): the optimizer and schedules start fresh."""
    path = ensure_local(path_or_name, base=base, token=token)
    return _load_light_weights(path, model)


def ensure_local(path_or_name: Union[str, Path], dest_dir: Union[str, Path] = ".",
                 base: Optional[str] = None, token: Optional[str] = None) -> Path:
    """Return a local path for ``path_or_name``, downloading from the relay if needed."""
    p = Path(path_or_name).expanduser()
    if p.is_file():
        return p
    rc = _relay()
    local = Path(dest_dir) / p.name
    local.parent.mkdir(parents=True, exist_ok=True)
    rc.download(str(path_or_name), str(local), base=base, token=token)
    return local


def _load_light_weights(path: Union[str, Path], model) -> Dict[str, Any]:
    d = torch.load(Path(path), map_location="cpu", weights_only=False)
    if model is not None:
        unwrap(model).load_state_dict(d["model"], strict=True)
    return d


# ------------------------------------------------------------------ Checkpointer

class Checkpointer:
    """Orchestrates periodic resume saves, final artifacts and uploads.

    Rank 0 writes; every rank may ``restore``.
    """

    def __init__(self, ckpt_dir: Union[str, Path], run_name: str, model,
                 optimizer=None, *, keep_n: int = 2, upload: bool = True,
                 relay_base: Optional[str] = None, relay_token: Optional[str] = None,
                 sigterm: bool = True, rank: int = 0,
                 final_upload_max_s: float = 2700.0, run_id: Optional[str] = None,
                 tokenizer_sha256: Optional[str] = None,
                 periodic_upload_max_s: float = 900.0, stop_upload_max_s: float = 1800.0):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_name = str(run_name)
        # the job's one run id (env KENCHI_RUN_ID on the node): embedded in
        # checkpoint names + metadata; auto-resume picks the newest
        # resume_<run_id>_* checkpoint the relay holds for it.
        self.run_id = str(run_id or run_name)
        # the tokenizer model sha (the job asserts it at start and refuses to
        # resume from a checkpoint whose tokenizer differs)
        self.tokenizer_sha256 = tokenizer_sha256
        self.model = model
        self.optimizer = optimizer
        self.keep_n = int(keep_n)
        self.upload_enabled = bool(upload)
        self.relay_base = relay_base
        self.relay_token = relay_token
        self.rank = int(rank)
        self.final_upload_max_s = float(final_upload_max_s)
        self.periodic_upload_max_s = float(periodic_upload_max_s)
        self.stop_upload_max_s = float(stop_upload_max_s)
        self._step = 0
        self._tokens = 0
        self._terminate = threading.Event()
        self._busy = threading.Event()   # set while a background save+upload runs
        if sigterm:
            self.arm_sigterm()

    # ------------------------------------------------------------ state
    def note_progress(self, step: int, tokens: int) -> None:
        """Record the trainer's current step/tokens (used by SIGTERM saves)."""
        self._step = int(step)
        self._tokens = int(tokens)

    def terminate_requested(self) -> bool:
        return self._terminate.is_set()

    def arm_sigterm(self) -> None:
        """Set ``terminate_requested`` on SIGTERM; saving happens in the trainer
        (no saving from signal-handler threads while the optimizer is running)."""
        def handler(signum, frame):  # pragma: no cover - signal path
            self._terminate.set()
        try:
            signal.signal(signal.SIGTERM, handler)
        except ValueError:  # not in the main thread (tests)
            pass

    # ------------------------------------------------------------ saving
    def save_resume(self, step: int, tokens: int, schedule=None, stream=None,
                    config=None, wait: bool = False,
                    trained_s: float = 0.0,
                    tokenizer_sha256: Optional[str] = None,
                    extra: Optional[Dict[str, Any]] = None):
        """A "resume" checkpoint: bf16 weights + bf16 AdamW state + step/tokens/
        schedule/stream/RNG state + run_id + trained_s (+ ``extra``: lr, recent
        loss).  Returns ``(path, delivered)``.

        Rank 0 snapshots to CPU at a step boundary.  ``wait=False`` (periodic):
        ONE background thread writes and uploads, weights first
        (``weights_<run>_<step>.pt``, usable alone), then the full resume file;
        each upload is verified by sha256 + size on the relay.  If the previous
        checkpoint is still uploading, this one is skipped and logged.
        ``wait=True`` (stop / SIGTERM): write and upload the resume file before
        returning; ``delivered`` says whether the relay confirmed it."""
        if self.rank != 0:
            return None, False
        if not wait and self._busy.is_set():
            print(f"CKPT_SKIPPED step={int(step)} (previous checkpoint still uploading)", flush=True)
            return None, False
        name = f"{self.run_id}_{int(step):012d}.pt"
        path = self.dir / f"resume_{name}"
        wpath = self.dir / f"weights_{name}"
        weights = _clones_cpu(unwrap(self.model), torch.bfloat16)
        head = {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": self.run_id,
            "step": int(step),
            "tokens": int(tokens),
            "trained_s": float(trained_s or 0.0),
            "tokenizer_sha256": tokenizer_sha256 or self.tokenizer_sha256,
            "config": config or {},
        }
        head.update(extra or {})
        snap = dict(head)
        snap.update({
            "format": RESUME_FORMAT,
            "model": weights,
            "optimizer": _clone_opt_state(self.optimizer, dtype=torch.bfloat16)
                         if self.optimizer is not None else None,
            "schedule": _state_of(schedule),
            "stream": _state_of(stream),
            "rng": rng_snapshot(),
            "meta": {"run": self.run_name, "run_id": self.run_id, "kind": "resume"},
        })
        if wait:
            _write(path, snap)
            ok = self._upload_sync(path, max_s=self.stop_upload_max_s, tag="CKPT")
            return path, ok

        def run():
            try:
                light = dict(head)
                light.update({"format": LIGHT_FORMAT, "model": weights})
                _write(wpath, light)
                self._upload_sync(wpath, max_s=self.periodic_upload_max_s, tag="CKPT")
                with_suppress(wpath.unlink)
                _write(path, snap)
                self._upload_sync(path, max_s=self.periodic_upload_max_s, tag="CKPT")
                self._rotate("resume")
            except Exception as e:  # noqa: BLE001 -- periodic saves are best-effort
                print(f"CKPT_FAILED step={int(step)} {e}", flush=True)
            finally:
                self._busy.clear()
        self._busy.set()
        threading.Thread(target=run, name="kenchi-ckpt-save", daemon=True).start()
        return path, False

    def final_uploads(self, run_dir: Union[str, Path], config=None, schedule=None,
                      stream=None, log_path=None, tokenizer_path=None,
                      with_resume: bool = False) -> Dict[str, Any]:
        """Final artifacts, uploaded one after another and each verified by
        sha256 + size on the relay: (1) ``<run>_final.safetensors`` (fp32),
        (2) ``<run>_final_config.json``, (3) ``<run>_final_tokenizer.model``,
        (4) the metrics log, and only ``with_resume``: (5) an fp32 resume
        checkpoint.  All of them share one retry budget (``final_upload_max_s``).
        Prints a DELIVERED / NOT_DELIVERED line per artifact and returns
        ``{"delivered": [...], "not_delivered": [...]}``."""
        out: Dict[str, Any] = {"delivered": [], "not_delivered": []}
        if self.rank != 0:
            return out
        run_dir = Path(run_dir)
        deadline = time.time() + self.final_upload_max_s
        # wait for a periodic checkpoint upload still in flight (bounded), so
        # the two do not share the link
        t_wait = time.time()
        while self._busy.is_set() and time.time() - t_wait < 600:
            time.sleep(2)

        def push(local: Optional[Path], remote: Optional[str] = None):
            if local is None or not Path(local).is_file():
                return
            remote = remote or Path(local).name
            ok = self._upload_sync(Path(local), remote, max_s=max(30.0, deadline - time.time()),
                                   tag="FINAL")
            out["delivered" if ok else "not_delivered"].append(remote)

        push(self._write_final_safetensors(run_dir / f"{self.run_id}_final.safetensors"))
        if config is not None:
            p = run_dir / f"{self.run_id}_final_config.json"
            p.write_text(json.dumps(config, default=str, indent=2))
            push(p)
        if tokenizer_path is not None:
            push(Path(tokenizer_path), f"{self.run_id}_final_tokenizer.model")
        if log_path is not None:
            push(Path(log_path))
        if with_resume:
            path = self.dir / f"resume_{self.run_id}_final.pt"
            save_full(path, self.model, self.optimizer, step=self._step, tokens=self._tokens,
                      schedule=schedule, stream=stream, config=config, wait=True,
                      tokenizer_sha256=self.tokenizer_sha256)
            push(path)
        return out

    def _write_final_safetensors(self, path: Path) -> Optional[Path]:
        sd = unwrap(self.model).state_dict()
        out, seen = {}, set()
        for k, v in sd.items():
            if v.data_ptr() in seen:  # tied weights (lm_head == tok_emb)
                continue
            seen.add(v.data_ptr())
            out[k] = v.detach().to("cpu", torch.float32).contiguous().clone()
        try:
            from safetensors.torch import save_file
            save_file(out, str(path), metadata={"format": "pt", "run_id": self.run_id,
                                                "step": str(self._step), "tokens": str(self._tokens),
                                                "tokenizer_sha256": str(self.tokenizer_sha256)})
        except ImportError:
            torch.save(out, path)
            print(f"WARNING safetensors not installed; wrote {path.name} with torch.save", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR could not write {path}: {e}", flush=True)
            return None
        return path

    # ------------------------------------------------------------ uploads
    def _upload_sync(self, local: Path, remote: Optional[str] = None,
                     max_s: Optional[float] = None, tag: str = "FINAL") -> bool:
        """Upload and verify (sha256 + size confirmed by the relay), retrying for
        at most ``max_s`` seconds.  Prints ``DELIVERED <name> ...`` or
        ``NOT_DELIVERED <name> ...``; never raises."""
        remote = remote or Path(local).name
        if not self.upload_enabled:
            return True
        max_s = float(self.final_upload_max_s if max_s is None else max_s)
        try:
            rc = _relay()
            token = self.relay_token or rc.resolve_token()
            urls = [self.relay_base] if self.relay_base else rc.resolve_urls()
            size = Path(local).stat().st_size
        except Exception as e:  # noqa: BLE001
            print(f"NOT_DELIVERED {remote} kind={tag} error={e}", flush=True)
            return False
        if not token or not urls:
            print(f"NOT_DELIVERED {remote} kind={tag} error=no relay configured", flush=True)
            return False
        t0 = time.time()
        deadline = t0 + max_s
        attempt, last = 0, None
        done = threading.Event()

        def heartbeat():   # a long upload must not look like a hang
            while not done.wait(60):
                print(f"UPLOADING {remote} kind={tag} size={size} elapsed={time.time() - t0:.0f}s",
                      flush=True)

        threading.Thread(target=heartbeat, daemon=True).start()
        try:
            return self._upload_loop(rc, local, remote, tag, size, t0, deadline)
        finally:
            done.set()

    def _upload_loop(self, rc, local, remote, tag, size, t0, deadline) -> bool:
        attempt, last = 0, None
        while True:
            attempt += 1
            try:
                stats = rc.upload(str(local), remote, base=self.relay_base,
                                  token=self.relay_token, retries=3)
                print(f"DELIVERED {remote} kind={tag} size={stats['size']} "
                      f"sha256={stats.get('sha256')} seconds={time.time() - t0:.0f} "
                      f"attempts={attempt} via={stats.get('url')}", flush=True)
                return True
            except Exception as e:  # noqa: BLE001
                last = e
                if time.time() >= deadline:
                    break
                time.sleep(min(15.0, max(1.0, deadline - time.time())))
        print(f"NOT_DELIVERED {remote} kind={tag} size={size} after={time.time() - t0:.0f}s "
              f"attempts={attempt} error={last}", flush=True)
        return False

    # ------------------------------------------------------------ rotation
    def _rotate(self, kind: str) -> None:
        files = checkpoints_on_disk(self.dir, kind, self.run_id)
        for p in files[:-self.keep_n]:
            try:
                p.unlink()
                log.info("[ckpt] rotated out %s", p.name)
            except OSError as e:  # noqa: PERF203
                log.warning("[ckpt] could not delete %s: %s", p, e)

    # ------------------------------------------------------------ restore
    def restore(self, path: Union[str, Path], *, model=None, optimizer=None,
                stream=None, schedule=None, rank: Optional[int] = None) -> Dict[str, Any]:
        """Apply a full/resume checkpoint: model + optimizer; return the raw dict.

        ``path`` may be a relay name (downloaded first).
        """
        path = ensure_local(path, self.dir, base=self.relay_base, token=self.relay_token)
        d = load_full(path, model=model or self.model,
                      optimizer=optimizer or self.optimizer)
        if stream is not None and d.get("stream") is not None:
            stream.load_state(d["stream"], rank=rank)
        if schedule is not None and d.get("schedule") is not None:
            schedule.load_state(d["schedule"])
        rng_restore(d.get("rng") or {})
        return d
