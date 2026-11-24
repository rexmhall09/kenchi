"""Phase 2 data stream (src/kenchi/shard_stream.py).

Streams pre-tokenized shards -- the modded-nanogpt `.bin` layout, see
``kenchi.data`` (magic 20240520, 256-int32 header, uint16 token stream) -- as
``(x, y)`` windows of shape ``(batch_size, seq_len)``.  ``y[:, t] == x[:, t+1]``
and the last column carries the teacher-forced target one step past the window
(the same convention as ``kenchi.data.ShardLoader``).

Design (docs-free summary of the contract):

- **Sources** are named groups of shard files with mixing weights, e.g.
  ``{"fineweb_edu": {"pattern": "data/fwe/*.bin", "weight": 0.85}}``.
- **Phases**: the config has an ordered list of phases, each with a token count
  (or a fraction of ``total_tokens``) and its own source weights -- how the
  final "anneal" on a higher-quality mix is expressed.
- **Row picking**: each batch row is drawn from a source with probability
  proportional to its weight; the draw is seeded per step from
  ``(seed, step, rank)`` so it is deterministic given those.
- **Rank lanes**: within a source, shards are read in a fixed order
  (sorted by path); the stream keeps a per-rank cursor that starts at ``rank``
  and advances by ``world_size`` per row, so the ranks of a DDP job read
  disjoint window slices of the same source stream.
- **Availability**: the stream reads only files that exist in the final name
  (the downloader writes ``*.part`` and renames on completion, see
  ``ShardDownloader`` below).  It never blocks: if the shard a cursor points at
  is not available yet, it wraps to an already-available shard and logs the
  event.
- **Memory**: at most ``max_shards_per_source`` shards per source are held in
  RAM per rank, loaded to pinned CPU memory when training on GPU and moved to
  the device per batch.

``ShardStream`` is the reader; ``ShardDownloader`` fetches new shards from a
manifest; ``wait_for_step0`` is the CLI mode the job entry script uses to block
until the shards needed for the first optimizer step exist.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from kenchi import data as kdata

log = logging.getLogger("kenchi.shard_stream")

__all__ = [
    "SourceSpec", "PhaseSpec", "ShardStream", "ShardDownloader",
    "wait_for_step0_from_config", "main",
]

MAX_SHARDS_PER_SOURCE = 2
DOWNLOAD_RETRIES = 4
DOWNLOAD_TIMEOUT_S = 60.0


# ------------------------------------------------------------------ config

@dataclass
class SourceSpec:
    """A named group of shards with a mixing weight."""

    name: str
    patterns: List[str] = field(default_factory=list)
    weight: float = 1.0

    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> "SourceSpec":
        pats = d.get("pattern")
        if pats is None:
            pats = d.get("patterns")
        if isinstance(pats, str):
            pats = [pats]
        elif pats is None:
            pats = []
        return cls(name=name, patterns=list(pats), weight=float(d.get("weight", 1.0)))

    def shard_paths(self, base_dir: Union[str, Path] = ".") -> List[Path]:
        """Sorted, complete (non-``.part``) shard paths matching this source's patterns."""
        base = Path(base_dir)
        paths: List[Path] = []
        for pat in self.patterns:
            p = Path(pat)
            full = p if p.is_absolute() else base / p
            if full.is_dir():
                paths.extend(sorted(full.glob("*")))
            else:
                paths.extend(sorted(full.parent.glob(full.name)))
        out = []
        for p in paths:
            if not p.is_file():
                continue
            if str(p).endswith(".part"):
                continue
            if p not in out:
                out.append(p)
        return sorted(out)


@dataclass
class PhaseSpec:
    """One phase of the schedule: token budget (or fraction of total) + weights."""

    tokens: Optional[int] = None
    fraction: Optional[float] = None
    weights: Optional[Dict[str, float]] = None
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PhaseSpec":
        return cls(
            tokens=(int(d["tokens"]) if d.get("tokens") is not None else None),
            fraction=(float(d["fraction"]) if d.get("fraction") is not None else None),
            weights=(dict(d["weights"]) if d.get("weights") is not None else None),
            name=d.get("name"),
        )


def sources_from_dict(d: Mapping[str, Any]) -> Dict[str, SourceSpec]:
    return {name: SourceSpec.from_dict(name, spec) for name, spec in (d or {}).items()}


def phases_from_list(lst: Sequence[Mapping[str, Any]]) -> List[PhaseSpec]:
    return [PhaseSpec.from_dict(p) for p in (lst or [])]


# ------------------------------------------------------------------ helpers

def _stable_seed(*parts) -> int:
    """A deterministic 63-bit seed from the given parts (no PYTHONHASHSEED dependence)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def _shard_n_tokens(path: Path) -> int:
    """Token count of a shard from its header; 0 when unreadable."""
    try:
        return int(kdata.read_shard_header(path)[2])
    except Exception:  # noqa: BLE001 -- a half-downloaded or corrupt file is not data
        return 0


# ------------------------------------------------------------------ the stream

class ShardStream:
    """Deterministic, rank-partitioned, phase-aware stream of training windows.

    ``next_batch()`` returns ``(x, y)`` of shape ``(batch_size, seq_len)`` on
    ``device``; each call reads ``batch_size`` rows, drawing each row's source
    from the current phase's weights (seeded by ``seed``/``rank``/step).
    """

    def __init__(self,
                 sources: Mapping[str, SourceSpec],
                 phases: Sequence[PhaseSpec],
                 seq_len: int,
                 batch_size: int,
                 *,
                 rank: int = 0,
                 world_size: int = 1,
                 device: Union[str, torch.device] = "cpu",
                 seed: int = 0,
                 base_dir: Union[str, Path] = ".",
                 total_tokens: int = 0,
                 available: Optional[Union[Callable[[str], List[Path]],
                                           Mapping[str, List[Path]]]] = None,
                 max_shards_per_source: int = MAX_SHARDS_PER_SOURCE):
        if not sources:
            raise ValueError("ShardStream needs at least one source")
        if int(seq_len) < 2:
            raise ValueError("seq_len must be >= 2")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        if int(world_size) < 1:
            raise ValueError("world_size must be >= 1")
        self.sources: Dict[str, SourceSpec] = dict(sources)
        self.phases: List[PhaseSpec] = list(phases) or [PhaseSpec()]
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.base_dir = Path(base_dir)
        self.total_tokens = int(total_tokens or 0)
        self.device = torch.device(device)
        self.max_shards_per_source = int(max_shards_per_source)

        # phase bounds: self._bounds[i] is the cumulative token end of phase i
        self._bounds: List[int] = []
        cum = 0
        for ph in self.phases:
            if ph.tokens is not None:
                b = int(ph.tokens)
            elif ph.fraction is not None:
                b = int(round(float(ph.fraction) * self.total_tokens))
            else:
                b = None
            if b is None:
                b = max(1, int(self.total_tokens) - cum)
            cum += b
            self._bounds.append(cum)
        if self.total_tokens > 0 and cum < self.total_tokens:
            self._bounds[-1] += self.total_tokens - cum
        if not self._bounds:
            self._bounds = [max(1, self.total_tokens)]

        # per-source default weights = the sources' weights
        self._default_weights: Dict[str, float] = {
            name: float(spec.weight) for name, spec in self.sources.items()}
        self._phase_weights: List[Dict[str, float]] = []
        for i, ph in enumerate(self.phases):
            w = dict(ph.weights) if ph.weights else dict(self._default_weights)
            w = {k: float(v) for k, v in w.items() if v > 0}
            if not w:
                w = dict(self._default_weights)
            self._phase_weights.append(w)
        self._phase_idx = 0
        self._weights: Dict[str, float] = dict(self._phase_weights[0])

        # availability: callable(source) -> paths, or a manifest dict, or default glob
        if available is None:
            self._available_fn = self._default_available
        elif callable(available):
            self._available_fn = available
        else:
            self._available_fn = (lambda name: list(available.get(name, [])))
        self._avail: Dict[str, List[Path]] = {}

        # per-rank, per-source state: cursor = the next global window index to read
        self._state: Dict[str, Dict[str, int]] = {
            name: {"cursor": self.rank} for name in self.sources}
        self._seen_available: Dict[str, bool] = {name: False for name in self.sources}
        self._loaded: Dict[str, "OrderedDict[Path, torch.Tensor]"] = {
            name: OrderedDict() for name in self.sources}
        self._n_cache: Dict[Tuple[str, int], int] = {}

        self._step = 0
        self.events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ config views

    @property
    def phase_names(self) -> List[str]:
        return [ph.name or f"phase{i}" for i, ph in enumerate(self.phases)]

    def current_phase(self, tokens_done: int) -> int:
        """Index of the phase active at ``tokens_done``."""
        for i, b in enumerate(self._bounds):
            if tokens_done < b:
                return i
        return len(self._bounds) - 1

    def current_phase_name(self, tokens_done: Optional[int] = None) -> str:
        if tokens_done is None:
            return self.phase_names[self._phase_idx]
        return self.phase_names[self.current_phase(tokens_done)]

    def advance_phase_if_needed(self, tokens_done: int) -> bool:
        """Switch to the phase active at ``tokens_done``; log on a switch."""
        new = self.current_phase(tokens_done)
        if new != self._phase_idx:
            old = self._phase_idx
            self._phase_idx = new
            self._weights = dict(self._phase_weights[new])
            self.events.append({"event": "phase_switch", "from": old, "to": new,
                                "tokens": int(tokens_done),
                                "name": self.phase_names[new]})
            log.info("[shard_stream] phase switch %s -> %s at %d tokens",
                     self.phase_names[old], self.phase_names[new], tokens_done)
            return True
        return False

    def phase_for_progress(self, progress: float) -> int:
        """Index of the phase active at schedule progress ``progress`` (0..1).
        Phase lengths are fractions of the planned run, so when a time budget
        re-plans the total, the last phase is still the last N% of the run."""
        total = float(self._bounds[-1]) or 1.0
        for i, b in enumerate(self._bounds):
            if progress < b / total:
                return i
        return len(self._bounds) - 1

    def advance_phase_by_progress(self, progress: float, tokens_done: int = 0) -> bool:
        """Switch to the phase active at schedule progress ``progress``."""
        new = self.phase_for_progress(float(progress))
        if new != self._phase_idx:
            old = self._phase_idx
            self._phase_idx = new
            self._weights = dict(self._phase_weights[new])
            self.events.append({"event": "phase_switch", "from": old, "to": new,
                                "tokens": int(tokens_done), "progress": float(progress),
                                "name": self.phase_names[new]})
            log.info("[shard_stream] phase switch %s -> %s at progress %.4f (%d tokens)",
                     self.phase_names[old], self.phase_names[new], progress, tokens_done)
            return True
        return False

    # ------------------------------------------------------------ availability

    def _default_available(self, name: str) -> List[Path]:
        return self.sources[name].shard_paths(self.base_dir)

    def refresh_available(self) -> Dict[str, List[Path]]:
        """Re-ask the availability source for each stream source; log when a
        source becomes available (sources join as their first shard arrives)."""
        self._avail = {name: [Path(p) for p in self._available_fn(name)]
                       for name in self.sources}
        for name, paths in self._avail.items():
            nwins = [self._n_win_of(name, i) for i in range(len(paths))]
            if any(nwins) and not self._seen_available.get(name):
                self._seen_available[name] = True
                log.info("[shard_stream] source %r now available (%d shards, %d windows)",
                         name, len(paths), sum(nwins))
        return self._avail

    def has_step0_data(self, require_all: bool = False) -> bool:
        """True when phase-0 sources have >= 1 usable shard (n_win >= 1).

        ``require_all=False`` (the entry script default): start as soon as ANY
        phase-0 source has data -- sources still downloading join later (the
        stream logs when they do)."""
        names = [n for n, w in self._phase_weights[0].items() if w > 0] or list(self.sources)
        avail = self.refresh_available()

        def _usable(name):
            return any(self._n_win_of(name, i) > 0 for i in range(len(avail.get(name, []))))
        if require_all:
            return all(_usable(n) for n in names)
        return any(_usable(n) for n in names)

    # ------------------------------------------------------------ shard access

    def _n_tokens_of(self, name: str, idx: int) -> int:
        key = (name, idx)
        if key not in self._n_cache:
            self._n_cache[key] = _shard_n_tokens(self._avail[name][idx])
        return self._n_cache[key]

    def _n_win_of(self, name: str, idx: int) -> int:
        n = self._n_tokens_of(name, idx)
        # window starts, as in kenchi.data.ShardLoader: arange(0, n - T, T)
        return len(np.arange(0, n - self.seq_len, self.seq_len))

    def _tokens(self, name: str, idx: int) -> torch.Tensor:
        path = self._avail[name][idx]
        tok = self._loaded[name].get(path)
        if tok is None:
            tok = kdata.load_shard(path, "cpu")
            if self.device.type == "cuda":
                tok = tok.pin_memory()
            self._loaded[name][path] = tok
            self._loaded[name].move_to_end(path)
            while len(self._loaded[name]) > self.max_shards_per_source:
                self._loaded[name].popitem(last=False)
        return tok

    def _locate(self, name: str, g: int) -> Optional[Tuple[int, int]]:
        """(shard_idx, local_window) of the available shard covering global window
        ``g``; None when the cursor is not covered by any available shard."""
        base = 0
        for idx in range(len(self._avail[name])):
            nw = self._n_win_of(name, idx)
            if nw > 0 and base <= g < base + nw:
                return idx, g - base
            base += nw
        return None

    def _wrap(self, name: str, g: int) -> Optional[Tuple[int, int, int]]:
        """(shard_idx, local_window, new_cursor) when wrapping to already-available
        shards; the new cursor keeps the rank's lane (cursor % world == rank)."""
        base = 0
        for idx in range(len(self._avail[name])):
            nw = self._n_win_of(name, idx)
            if nw <= 0:
                continue
            local0 = (self.rank - base) % self.world_size
            if local0 < nw:
                new_c = base + local0
                self.events.append({"event": "wrap", "source": name,
                                    "cursor": int(g), "new_cursor": int(new_c)})
                log.warning("[shard_stream] shard for cursor %d not available; "
                            "wrapping to available shard %s window %d",
                            g, self._avail[name][idx], local0)
                return idx, local0, new_c
            base += nw
        return None

    def _read_row(self, name: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        st = self._state[name]
        g = st["cursor"]
        loc = self._locate(name, g)
        if loc is None:
            wrapped = self._wrap(name, g)
            if wrapped is None:
                return None
            idx, local, g = wrapped
            st["cursor"] = g
        else:
            idx, local = loc
        tokens = self._tokens(name, idx)
        start = local * self.seq_len
        n = tokens.numel()
        if start + self.seq_len + 1 > n:
            return None  # cannot happen: _n_win_of guarantees the window fits
        st["cursor"] = g + self.world_size
        x = tokens[start:start + self.seq_len]
        y = tokens[start + 1:start + self.seq_len + 1]
        return x, y

    # ------------------------------------------------------------ batches

    def _active_sources(self) -> List[str]:
        names = [n for n, w in self._weights.items() if w > 0 and n in self._state]
        return names or list(self.sources)

    def _pick_source(self, rng: torch.Generator, names: Sequence[str]) -> str:
        w = torch.tensor([self._weights[n] for n in names], dtype=torch.float32)
        i = int(torch.multinomial(w, 1, generator=rng).item())
        return names[i]

    def next_batch(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """One ``(x, y)`` batch of ``batch_size`` rows, shape ``(B, seq_len)``."""
        b = int(batch_size if batch_size is not None else self.batch_size)
        if not self._avail or self._step % 200 == 0:
            self.refresh_available()   # the job has all shards before training starts
        names = self._active_sources()
        if not names:
            raise RuntimeError("ShardStream: no sources configured")
        rng = torch.Generator().manual_seed(
            _stable_seed("shard_stream", self.seed, self.rank, self._step))
        rows: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(b):
            row = None
            for _attempt in range(4 * len(names) + 4):
                src = self._pick_source(rng, names)
                row = self._read_row(src)
                if row is not None:
                    break
            if row is None:
                raise RuntimeError(
                    f"ShardStream: no usable shards for sources {names} "
                    f"(available: { {n: [p.name for p in v] for n, v in self._avail.items()} })")
            rows.append(row)
        self._step += 1
        x = torch.stack([r[0] for r in rows]).to(self.device, non_blocking=True)
        y = torch.stack([r[1] for r in rows]).to(self.device, non_blocking=True)
        return x, y

    # ------------------------------------------------------------ state

    def state(self) -> Dict[str, Any]:
        """Checkpointable position: step, phase, per-source rank cursors."""
        return {
            "step": int(self._step),
            "phase_idx": int(self._phase_idx),
            "cursors": {name: int(st["cursor"]) for name, st in self._state.items()},
        }

    def load_state(self, state: Mapping[str, Any], rank: Optional[int] = None) -> None:
        """Restore a ``state()`` snapshot. ``rank`` aligns cursors to the lane of the
        rank resuming (each rank's cursor stays congruent to its lane mod world)."""
        self._step = int(state.get("step", self._step))
        ph = int(state.get("phase_idx", 0))
        if 0 <= ph < len(self._bounds):
            self._phase_idx = ph
            self._weights = dict(self._phase_weights[ph])
        for name, c in (state.get("cursors") or {}).items():
            if name not in self._state:
                continue
            c = int(c)
            if rank is not None and self.world_size > 1:
                c = c + ((rank - c) % self.world_size)
            self._state[name]["cursor"] = c


# ------------------------------------------------------------------ downloader

@dataclass
class ShardEntry:
    url: str
    dest: str
    source: str
    size: Optional[int] = None


def _import_relay():
    """Import ``relay_client`` from the training helper directory."""
    root = Path(__file__).resolve().parents[1]
    relay_dir = root / "train"
    if relay_dir.is_dir() and str(relay_dir) not in sys.path:
        sys.path.insert(0, str(relay_dir))
    import relay_client  # noqa: WPS433 -- local import by layout
    return relay_client


def _relay_urls() -> List[str]:
    """The configured relay base URLs (env or the supervisor's file); empty when
    none are configured (downloads then use ``relay_client``'s own resolution)."""
    try:
        rc = _import_relay()
        return [u for u in rc.resolve_urls() if u]
    except Exception:  # noqa: BLE001
        return []


def _fetch_http(url: str, part: Path, timeout: float, retries: int) -> int:
    import urllib.request
    last = None
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kenchi"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            return part.stat().st_size
        except Exception as e:  # noqa: BLE001 -- retried below
            last = e
            log.warning("[shard_downloader] %s attempt %d failed: %s", url, attempt, e)
            time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    raise RuntimeError(f"download failed: {url}: {last}")


class ShardDownloader:
    """Fetches shard URLs listed in a manifest JSON with N parallel workers.

    Manifest format: ``{"shards": [{"url": ..., "dest": ..., "source": ...,
    "size": optional}]}``.  Downloads write to ``<dest>.part`` and are renamed
    into place on completion; ``relay://`` URLs are fetched through
    ``relay_client`` (the workspace machine's checkpoint relay).
    """

    def __init__(self, manifest_path: Union[str, Path], workers: int = 8,
                 timeout: float = DOWNLOAD_TIMEOUT_S, retries: int = DOWNLOAD_RETRIES):
        import json
        manifest = json.loads(Path(manifest_path).read_text())
        if isinstance(manifest, list):
            raw = manifest
        elif isinstance(manifest, dict):
            raw = manifest.get("shards") or []
        else:
            raw = []
        self.entries = [ShardEntry(url=e["url"], dest=e["dest"], source=e.get("source", ""),
                                   size=e.get("size")) for e in raw]
        if not self.entries:
            raise ValueError(f"manifest {manifest_path} lists no shards")
        self.workers = int(workers)
        self.timeout = float(timeout)
        self.retries = int(retries)
        self._futs: List["as_completed"] = []
        self._done = threading.Event()
        self._n_ok = 0
        self._n_err = 0
        self._fetch_n = 0
        # the relay URLs: parallel downloads alternate between the tunnels
        self._relay_urls = _relay_urls()

    def _fetch(self, e: ShardEntry) -> None:
        try:
            self._fetch_one(e)
        except Exception as exc:  # noqa: BLE001 -- logged, counted, never fatal
            self._n_err += 1
            log.error("[shard_downloader] failed %s: %s", e.dest, exc)
            part = Path(str(e.dest) + ".part")
            try:
                part.unlink()
            except OSError:
                pass

    def _fetch_one(self, e: ShardEntry) -> None:
        dest = Path(e.dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = Path(str(dest) + ".part")
        if dest.exists():
            log.info("[shard_downloader] %s already present, skipping", dest)
            self._n_ok += 1
            return
        if e.url.startswith("relay://"):
            # relay_client downloads via its own <dest>.part and renames to dest;
            # alternate the tunnels per entry so parallel downloads use both
            rc = _import_relay()
            remote = e.url[len("relay://"):] or dest.name
            base = None
            if self._relay_urls:
                base = self._relay_urls[self._fetch_n % len(self._relay_urls)]
                self._fetch_n += 1
            rc.download(remote, str(dest), base=base, timeout=self.timeout,
                        retries=self.retries)
            size = dest.stat().st_size
        else:
            size = _fetch_http(e.url, part, self.timeout, self.retries)
            os.replace(part, dest)
        if e.size is not None and size != int(e.size):
            raise RuntimeError(f"size mismatch: expected {e.size}, got {size}")
        self._n_ok += 1
        log.info("[shard_downloader] done %s (%d bytes) from %s", dest, size, e.url)

    def start(self) -> None:
        def run():
            with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
                futs = [ex.submit(self._fetch, e) for e in self.entries]
                remaining = list(futs)
                while remaining:
                    remaining = [f for f in remaining if not f.done()]
                    print(f"[shard_downloader] progress: {self._n_ok}/{len(self.entries)} "
                          f"done, {self._n_err} failed, {len(remaining)} remaining",
                          flush=True)
                    if remaining:
                        time.sleep(20)
            self._done.set()
        t = threading.Thread(target=run, name="kenchi-shard-downloader", daemon=True)
        t.start()

    def wait_all(self, timeout: Optional[float] = None) -> bool:
        """Block until every entry is done; True when all succeeded."""
        self._done.wait(timeout)
        return self._n_err == 0


# ------------------------------------------------------------------ wait mode

def wait_for_step0_from_config(config_path: Union[str, Path], timeout: float = 900.0,
                               poll_s: float = 5.0, require_all: bool = False) -> bool:
    """Block until the shards needed for step 0 are available (see
    `main --wait-for-step0`).  With ``require_all=False`` (the entry script
    default) it returns as soon as ANY phase-0 source has usable data -- sources
    that are still downloading join later and are logged by the stream."""
    import json
    raw = json.loads(Path(config_path).read_text())
    sources = sources_from_dict(raw.get("sources") or {})
    phases = phases_from_list(raw.get("phases") or [])
    model_cfg = raw.get("model") or {}
    base_dir = raw.get("data_base_dir") or raw.get("base_dir") or "."
    if not sources:
        # fall back to the Phase 1 shard dir convention
        sources = {"shards": SourceSpec(name="shards", patterns=["data/shards/*.bin"],
                                        weight=1.0)}
        phases = phases or [PhaseSpec()]
    stream = ShardStream(sources=sources, phases=phases or [PhaseSpec()],
                         seq_len=int(model_cfg.get("seq_len", 1024)), batch_size=1,
                         total_tokens=int(raw.get("total_tokens") or 0),
                         base_dir=base_dir)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if stream.has_step0_data(require_all=require_all):
            log.info("[shard_stream] step-0 shards available after %.0f s", time.time() - t0)
            print(f"[shard_stream] step-0 shards available after {time.time() - t0:.0f} s",
                  flush=True)
            return True
        time.sleep(poll_s)
    log.warning("[shard_stream] timed out waiting for step-0 shards; proceeding anyway")
    print("[shard_stream] WARNING: timed out waiting for step-0 shards; proceeding anyway",
          flush=True)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kenchi.shard_stream",
                                 description="Phase 2 shard downloader / step-0 waiter")
    ap.add_argument("--manifest", help="manifest JSON with {'shards': [{url, dest, source, size}]}")
    ap.add_argument("--workers", type=int, default=8, help="parallel download workers")
    ap.add_argument("--wait-for-step0", action="store_true",
                    help="block until the shards needed for step 0 exist, then exit 0")
    ap.add_argument("--config", help="phase 2 config JSON (with sources/phases)")
    ap.add_argument("--timeout", type=float, default=900.0, help="wait-for-step0 timeout, s")
    args = ap.parse_args(argv)

    if args.wait_for_step0:
        if not args.config:
            ap.error("--wait-for-step0 requires --config")
        wait_for_step0_from_config(args.config, timeout=args.timeout)
        # proceed even on timeout: the stream wraps to already-available shards
        return 0

    if not args.manifest:
        ap.error("--manifest is required without --wait-for-step0")
    dl = ShardDownloader(args.manifest, workers=args.workers)
    print(f"[shard_downloader] {len(dl.entries)} entries from {args.manifest}; "
          f"starting {dl.workers} workers", flush=True)
    dl.start()
    while not dl._done.is_set():  # heartbeat until interrupted
        time.sleep(60)
        print(f"[shard_downloader] ok={dl._n_ok} failed={dl._n_err}", flush=True)
    print(f"[shard_downloader] finished: ok={dl._n_ok} failed={dl._n_err}", flush=True)
    return 0 if dl._n_err == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
