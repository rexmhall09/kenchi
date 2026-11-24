"""Shard IO and the deterministic batch loader for Phase 1 (docs/PHASE1_DESIGN.md §4).

Shard format: the modded-nanogpt `.bin` layout -- a 256-int32 header whose entries are
[0]=20240520 magic, [1]=1 version, [2]=number of tokens -- followed by the token stream
as uint16. Tokens are read fully into a torch tensor on the target device.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def write_shard(path, tokens) -> int:
    """Write `tokens` to a `.bin` shard in the modded-nanogpt format. Returns the token count."""
    if isinstance(tokens, torch.Tensor):
        arr = tokens.detach().to("cpu").numpy().astype(np.uint16, copy=False)
    else:
        arr = np.asarray(tokens, dtype=np.uint16)
    arr = np.ascontiguousarray(arr.reshape(-1))
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = arr.shape[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())
    return int(arr.shape[0])


def read_shard_header(path) -> Tuple[int, int, int]:
    """Return (magic, version, n_tokens) of a `.bin` shard."""
    with open(path, "rb") as f:
        header = np.frombuffer(f.read(HEADER_INTS * 4), dtype=np.int32)
    if header.shape[0] != HEADER_INTS:
        raise ValueError(f"{path}: shard header must be {HEADER_INTS} int32 entries")
    magic, version, n = int(header[0]), int(header[1]), int(header[2])
    if magic != SHARD_MAGIC:
        raise ValueError(f"{path}: bad magic {magic} (expected {SHARD_MAGIC})")
    if version != SHARD_VERSION:
        raise ValueError(f"{path}: unsupported version {version} (expected {SHARD_VERSION})")
    return magic, version, n


def load_shard(path, device="cpu") -> torch.Tensor:
    """Load a `.bin` shard fully into a 1-D int64 tensor of tokens on `device`."""
    path = Path(path)
    with open(path, "rb") as f:
        _, _, n = read_shard_header_from_file(f, path)
        data = np.frombuffer(f.read(), dtype=np.uint16, count=n)
    if data.shape[0] != n:
        raise ValueError(f"{path}: expected {n} tokens in the header, found {data.shape[0]}")
    return torch.from_numpy(data.astype(np.int64)).to(device)


def read_shard_header_from_file(f, path):
    header = np.frombuffer(f.read(HEADER_INTS * 4), dtype=np.int32)
    if header.shape[0] != HEADER_INTS:
        raise ValueError(f"{path}: shard header must be {HEADER_INTS} int32 entries")
    return int(header[0]), int(header[1]), int(header[2])


def write_random_shards(dir, n_shards, n_tokens, vocab_size=50304, seed=0,
                        name_prefix="shard") -> List[Path]:
    """Write `n_shards` random-token shards of `n_tokens` tokens each. For tests."""
    g = torch.Generator().manual_seed(int(seed))
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(int(n_shards)):
        toks = torch.randint(0, int(vocab_size), (int(n_tokens),), generator=g)
        p = dir / f"{name_prefix}_{i:04d}.bin"
        write_shard(p, toks)
        paths.append(p)
    return paths


class ShardLoader:
    """Deterministic (x, y) window batches from a list of pre-tokenized shards.

    A shard is loaded fully into a tensor on `device`. Windows of length T are the
    non-overlapping consecutive slices tokens[start:start+T] with start = 0, T, 2T,
    ..., up to the largest start where a full window still fits together with the
    teacher-forced target (start + T + 1 <= n). The window order is a seeded
    permutation per shard; when a shard is exhausted the loader moves to the next
    shard and, after the last shard, wraps around to the first again.

    `__iter__` yields infinite batches `(x, y)` of shape (B, T) where
    `y[:, t] == x[:, t + 1]` for t < T - 1 and `y[:, T - 1]` is the token right
    after the window.
    """

    def __init__(self, shard_paths: Sequence[Path], batch_size: int = 64, T: int = 1024,
                 device="cpu", seed: int = 0):
        self.shard_paths = [Path(p) for p in shard_paths]
        if not self.shard_paths:
            raise ValueError("ShardLoader needs at least one shard path")
        self.batch_size = int(batch_size)
        self.T = int(T)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.T < 2:
            raise ValueError("T must be >= 2")
        self.device = torch.device(device)
        self.seed = int(seed)

    def _window_starts(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.numel()
        if n < self.T + 1:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.arange(0, n - self.T, self.T, device=self.device)

    def _shard_batches(self, shard_idx: int) -> Iterator:
        path = self.shard_paths[shard_idx]
        tokens = load_shard(path, self.device)
        starts = self._window_starts(tokens)
        n_win = starts.numel()
        if n_win == 0:
            return
        # seeded permutation per shard
        g = torch.Generator(device="cpu").manual_seed(
            self.seed + 0x9E3779B9 * (shard_idx + 1) + 1)
        starts = starts[torch.randperm(n_win, generator=g).to(self.device)]
        T = self.T
        off = torch.arange(T, device=self.device)
        for b in range(0, n_win, self.batch_size):
            idx = starts[b:b + self.batch_size]
            if idx.numel() == 0:
                break
            x = tokens[idx[:, None] + off[None, :]]
            y = tokens[idx[:, None] + off[None, :] + 1]
            yield x, y

    def __iter__(self):
        while True:
            any_yielded = False
            for si in range(len(self.shard_paths)):
                for batch in self._shard_batches(si):
                    any_yielded = True
                    yield batch
            if not any_yielded:
                return


class FixedValSet:
    """Fixed validation subset: the first `n_tokens` tokens of a shard, in fixed order."""

    def __init__(self, shard_path, batch_size: int = 64, T: int = 1024, device="cpu",
                 n_tokens: int = 2_097_152):
        self.tokens = load_shard(shard_path, device)[: int(n_tokens)]
        self.batch_size = int(batch_size)
        self.T = int(T)
        self.device = torch.device(device)

    def __len__(self):
        return self.tokens.numel()

    def batches(self):
        tokens = self.tokens
        if tokens.numel() < self.T + 1:
            return
        starts = torch.arange(0, tokens.numel() - self.T, self.T, device=self.device)
        off = torch.arange(self.T, device=self.device)
        for b in range(0, starts.numel(), self.batch_size):
            idx = starts[b:b + self.batch_size]
            if idx.numel() == 0:
                break
            x = tokens[idx[:, None] + off[None, :]]
            y = tokens[idx[:, None] + off[None, :] + 1]
            yield x, y
