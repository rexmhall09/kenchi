"""Synthetic recall task (src/kenchi/recall_task.py), docs/PHASE1_DESIGN.md §5.

Multi-query associative recall with abstention, standard MQAR layout:

- Document: N pairs, 3 tokens per pair `k v1 v2` -- the key is a single token drawn
  from [V//8, V//4) and unique within the document; the value is two tokens drawn
  from [V//4, V//2).
- Queries `k a1 a2` immediately follow the document, until the row is full
  (n_q = (T - 3N) // 3). 75% of the queries use a present key -- drawn uniformly
  over the document's pairs, repeats allowed -- with a1 a2 = that pair's value;
  25% use a key from the key range that is NOT in the document, with a1 a2 =
  NONE NONE. No look-alike category.
- `loss_mask` is True at the two positions that predict a1 and a2 of every query:
  the query key position predicts a1; the a1 position predicts a2.

`RecallGenerator` is fully vectorised with torch ops (no Python loop per token) and
returns batches on `device`. `RecallEvaluator` scores models with the §5 metrics
per N (pooled over all queries) and reads signal values at the query key
positions for the uncertainty table of §6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional

import torch

from kenchi import metrics as kmetrics
from kenchi.metrics import dist_entropy

ABSENT_FRAC = 0.25                       # 25% of queries are absent (§5)
N_PAIR_MIN, N_PAIR_MAX = 2, 128          # N sampled uniformly in [2, 128]
PAD_ID = 0


class RecallIds(NamedTuple):
    """Recall token ids derived from the model vocab_size V (§5, MQAR layout).

    The K/V/Q/A specials stay reserved (unused): V-8..V-5. The one used special is
    NONE = V-4 (the abstain token). Keys are drawn from [V//8, V//4); values from
    [V//4, V//2) (the copy probe's ordinary range is [V//8, V//2)).
    """
    K_TOKEN: int       # reserved, unused (V - 8)
    V_TOKEN: int       # reserved, unused (V - 7)
    Q_TOKEN: int       # reserved, unused (V - 6)
    A_TOKEN: int       # reserved, unused (V - 5)
    NONE_TOKEN: int    # V - 4
    key_lo: int        # V // 8
    key_hi: int        # V // 4
    value_lo: int      # V // 4
    value_hi: int      # V // 2
    ordinary_lo: int   # V // 8
    ordinary_hi: int   # V // 2


def recall_ids(vocab_size: int) -> RecallIds:
    """The recall ids for a model vocab of size `vocab_size` (§5)."""
    v = int(vocab_size)
    if v < 32:
        raise ValueError(f"vocab_size {v} is too small for the recall specials")
    return RecallIds(
        K_TOKEN=v - 8, V_TOKEN=v - 7, Q_TOKEN=v - 6, A_TOKEN=v - 5, NONE_TOKEN=v - 4,
        key_lo=v // 8, key_hi=v // 4, value_lo=v // 4, value_hi=v // 2,
        ordinary_lo=v // 8, ordinary_hi=v // 2,
    )


@dataclass
class RecallBatch:
    """One generated batch of recall rows (all tensors on `device`).

    Query fields are (B, n_q_max); slots with `q_valid[b, q] == False` are padding
    (their key_pos is clamped to [0, T)).

    - tokens: (B, T): the document (N pairs `k v1 v2`), then the queries `k a1 a2`,
      then PAD.
    - targets: (B, T): targets[t] = tokens[t + 1].
    - loss_mask: (B, T) bool: True at the two positions that predict a1 and a2 of
      every query (the query key position predicts a1; the a1 position predicts a2).
    - key_pos: (B, n_q_max): position of the query key token of each query slot.
    - q_valid: (B, n_q_max) bool.
    - present: (B, n_q_max) bool (the query key is a document pair's key).
    - true_a1 / true_a2: (B, n_q_max) long -- the tokens the model should predict
      at key_pos and key_pos + 1 (the pair's value, or NONE for absent queries).
    """
    tokens: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor
    key_pos: torch.Tensor
    q_valid: torch.Tensor
    present: torch.Tensor
    true_a1: torch.Tensor
    true_a2: torch.Tensor


class RecallGenerator:
    """Vectorised generator of multi-query recall batches (one seeded CPU generator)."""

    def __init__(self, vocab_size: int = 50304, T: int = 1024, batch_size: int = 64,
                 device="cpu", seed: int = 0, n_min: int = N_PAIR_MIN,
                 n_max: int = N_PAIR_MAX, absent_frac: float = ABSENT_FRAC,
                 pad_id: int = PAD_ID):
        self.vocab_size = int(vocab_size)
        self.T = int(T)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.n_min = max(2, int(n_min))
        self.n_max = max(self.n_min, int(n_max))
        self.absent_frac = float(absent_frac)
        self.pad_id = int(pad_id)
        self.ids = recall_ids(vocab_size)
        self.gen = torch.Generator(device="cpu").manual_seed(self.seed)

    def next_batch(self, n_rows: Optional[int] = None) -> RecallBatch:
        ids = self.ids
        B = self.batch_size if n_rows is None else int(n_rows)
        if B < 1:
            raise ValueError("n_rows must be >= 1")
        T, g = self.T, self.gen
        N = torch.randint(self.n_min, self.n_max + 1, (B,), generator=g).clamp(min=2)
        n_max = int(N.max())
        if 3 * n_max + 3 > T:
            raise ValueError(
                f"T={T} is too small for N up to {n_max}: need T >= 3*N+3")
        key_range = ids.key_hi - ids.key_lo
        if key_range < n_max:
            raise ValueError(f"key range size {key_range} is too small for N up to {n_max}")
        tokens = torch.full((B, T), self.pad_id, dtype=torch.long)
        doc_valid = torch.arange(n_max).unsqueeze(0) < N.unsqueeze(1)      # (B, n_max)

        # ---- document: N pairs `k v1 v2` (3 tokens per pair); keys are unique
        #      within the document (argsort of random keys = sample without
        #      replacement), values are two tokens from [value_lo, value_hi)
        keys = (torch.rand(B, key_range, generator=g).argsort(dim=1)[:, :n_max]
                + ids.key_lo)                                               # (B, n_max)
        values = torch.randint(ids.value_lo, ids.value_hi, (B, n_max, 2), generator=g)
        pair_tok = torch.stack([keys, values[:, :, 0], values[:, :, 1]], dim=-1)  # (B, n_max, 3)
        pos = (3 * torch.arange(n_max).view(1, n_max, 1)
               + torch.arange(3).view(1, 1, 3)).expand(B, n_max, 3).reshape(B, n_max * 3)
        mask3 = doc_valid.unsqueeze(-1).expand(B, n_max, 3).reshape(B, n_max * 3)
        tokens.scatter_(1, pos, torch.where(mask3, pair_tok.reshape(B, n_max * 3),
                                            torch.full_like(pos, self.pad_id)))

        # ---- queries `k a1 a2`: n_q = (T - 3N) // 3 per row, right after the doc
        n_q = torch.div(T - 3 * N, 3, rounding_mode="floor").clamp(min=0)   # (B,)
        nq = int(n_q.max()) if B else 0
        qpos = 3 * N.unsqueeze(1) + 3 * torch.arange(nq).unsqueeze(0)        # (B, nq)
        key_pos = qpos
        q_valid = torch.arange(nq).unsqueeze(0) < n_q.unsqueeze(1)          # (B, nq)
        rows_q = torch.arange(B).unsqueeze(1).expand(B, nq)

        # 75% present: the key is a doc pair's key (uniform, repeats allowed)
        present = (torch.rand(B, nq, generator=g) >= self.absent_frac) & q_valid
        absent = q_valid & ~present
        src = torch.randint(0, n_max, (B, nq), generator=g) % N.unsqueeze(1)
        qkey = keys[rows_q, src]                                            # (B, nq)
        vals = values[rows_q, src]                                          # (B, nq, 2)
        true_a1 = torch.where(present, vals[..., 0],
                              torch.full_like(qkey, ids.NONE_TOKEN))
        true_a2 = torch.where(present, vals[..., 1],
                              torch.full_like(qkey, ids.NONE_TOKEN))

        # 25% absent: the key is from the key range but NOT in the document
        if absent.any():
            k = torch.randint(ids.key_lo, ids.key_hi, (B, nq), generator=g)
            collide = ((keys.unsqueeze(1) == k.unsqueeze(2)) & doc_valid.unsqueeze(1)
                       ).any(dim=2)                                         # (B, nq)
            for _ in range(16):
                if not collide.any():
                    break
                fix = collide
                k[fix] = torch.randint(ids.key_lo, ids.key_hi, (int(fix.sum()),),
                                       generator=g)
                collide = ((keys.unsqueeze(1) == k.unsqueeze(2))
                           & doc_valid.unsqueeze(1)).any(dim=2)
            qkey = torch.where(present, qkey, k)

        # ---- scatter the queries `k a1 a2` at qpos for valid slots
        qt = torch.stack([qkey, true_a1, true_a2], dim=-1)                  # (B, nq, 3)
        for o in range(3):
            pos = qpos + o
            m = q_valid
            tokens[rows_q[m], pos[m]] = qt[..., o][m]

        # ---- targets = tokens shifted left by one; loss_mask True at the two
        #      positions that predict a1 and a2 of every query
        targets = torch.roll(tokens, shifts=-1, dims=1)
        loss_mask = torch.zeros(B, T, dtype=torch.bool)
        for o in range(2):
            pos = (key_pos + o).clamp(0, T - 1)
            m = q_valid
            loss_mask[rows_q[m], pos[m]] = True
        key_pos = key_pos.clamp(0, T - 1)

        def _dev(t):
            return t.to(self.device)

        return RecallBatch(
            tokens=_dev(tokens),
            targets=_dev(targets),
            loss_mask=_dev(loss_mask),
            key_pos=_dev(key_pos),
            q_valid=_dev(q_valid),
            present=_dev(present),
            true_a1=_dev(true_a1),
            true_a2=_dev(true_a2),
        )


@dataclass
class RecallMetrics:
    """§5 metrics per N (pooled over all queries), plus per-query arrays for §6.

    - exact_match_present: fraction of present queries with both a1 and a2 right
      (a1: argmax(logits[key_pos]) == true_a1; a2: argmax(logits[key_pos + 1]) ==
      true_a2, the a1 position predicting a2).
    - abstention_accuracy: fraction of absent queries answered NONE at the key
      position (argmax(logits[key_pos]) == NONE).
    - false_abstention_rate: fraction of present queries answered NONE at the key
      position.
    - ece_first_answer: ECE of the confidence of the first answer token (max prob
      of the prediction at the key position vs first-token correctness).
    - n_queries: number of valid queries scored.
    - correct_a1: (n,) -- the model predicts the first answer token.
    - exact_match: (n,) -- both answer tokens right.
    - signals_at_key / entropy_at_key: per-query signal values read at the query
      key position.
    """
    N: int
    n_docs: int
    n_queries: int
    exact_match_present: float
    abstention_accuracy: Optional[float]
    false_abstention_rate: Optional[float]
    ece_first_answer: float
    correct_a1: torch.Tensor
    exact_match: torch.Tensor
    signals_at_key: Dict[str, torch.Tensor]
    entropy_at_key: torch.Tensor


class RecallEvaluator:
    """Scores models with the §5 recall metrics per N (docs/PHASE1_DESIGN.md §5).

    Documents are built at T = cfg.seq_len with fixed N in `n_values`; ALL queries
    are scored and pooled. The forward runs in row batches of `chunk_rows` under
    the same autocast context as training (`autocast_dtype`).
    """

    def __init__(self, model, cfg, device="cpu", n_values=(8, 32, 64, 128),
                 n_docs=8, seed=0, chunk_rows=16, autocast_dtype=None):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        T = int(cfg.seq_len)
        # "for short T use what fits": N with 3N+3 <= T
        self.n_values = tuple(N for N in n_values if 3 * int(N) + 3 <= T)
        self.n_docs = int(n_docs)
        self.seed = int(seed)
        self.chunk_rows = max(1, int(chunk_rows))
        self.autocast_dtype = autocast_dtype

    def evaluate(self, n_values=None) -> Dict[int, RecallMetrics]:
        vals = self.n_values if n_values is None else tuple(int(n) for n in n_values)
        return {N: self._evaluate_one(N) for N in vals}

    def _evaluate_one(self, N: int) -> RecallMetrics:
        T = int(self.cfg.seq_len)
        ids = recall_ids(self.cfg.vocab_size)
        gen = RecallGenerator(
            vocab_size=self.cfg.vocab_size, T=T, batch_size=self.n_docs,
            device=self.device, seed=self.seed + 17 * N, n_min=N, n_max=N)
        batch = gen.next_batch()
        tokens, key_pos, q_valid = batch.tokens, batch.key_pos, batch.q_valid
        B, nq = q_valid.shape
        present, absent = batch.present, ~batch.present
        true_a1, true_a2 = batch.true_a1, batch.true_a2
        pred_a1 = torch.zeros(B, nq, dtype=torch.long, device=self.device)
        pred_a2 = torch.zeros_like(pred_a1)
        conf_key = torch.zeros(B, nq, dtype=torch.float32, device=self.device)
        entropy_key = torch.zeros_like(conf_key)
        aux: Dict[str, torch.Tensor] = {}
        aux_out = {"u_tok_logit": "u_tok", "u_span_logit": "u_span",
                   "jepa_logvar": "jepa_logvar", "jepa_err": "jepa_err"}
        self.model.eval()
        with torch.no_grad():
            for c0 in range(0, B, self.chunk_rows):
                c1 = min(B, c0 + self.chunk_rows)
                out = self.model(tokens[c0:c1], targets=None, loss_mask=None,
                                 return_aux=True)
                logits = out["logits"]
                idx = key_pos[c0:c1].clamp(0, T - 1)
                m = q_valid[c0:c1]
                if not m.any():
                    continue
                rows_c = torch.arange(c1 - c0, device=self.device).unsqueeze(1).expand(c1 - c0, nq)
                p_key = torch.softmax(logits[rows_c[m], idx[m]].float(), dim=-1)
                p_a1 = torch.softmax(logits[rows_c[m], (idx[m] + 1).clamp(max=T - 1)].float(),
                                     dim=-1)
                pred_a1[c0:c1][m] = p_key.argmax(-1)
                pred_a2[c0:c1][m] = p_a1.argmax(-1)
                conf_key[c0:c1][m] = p_key.amax(-1)
                entropy_key[c0:c1][m] = dist_entropy(p_key)
                for out_key, name in aux_out.items():
                    if out_key in out and name not in aux:
                        aux[name] = torch.zeros(B, nq, dtype=torch.float32,
                                                device=self.device)
                    if name in aux:
                        v = out[out_key][rows_c[m], idx[m]].float()
                        if out_key in ("u_tok_logit", "u_span_logit"):
                            v = torch.sigmoid(v)
                        aux[name][c0:c1][m] = v
                if "loop_delta" in out and "loop_delta_last" not in aux:
                    aux["loop_delta_last"] = torch.zeros(
                        B, nq, dtype=torch.float32, device=self.device)
                if "loop_delta_last" in aux:
                    aux["loop_delta_last"][c0:c1][m] = out["loop_delta"][rows_c[m], idx[m], -1].float()

        v = q_valid
        a1_ok = pred_a1[v] == true_a1[v]
        exact_match = a1_ok & (pred_a2[v] == true_a2[v])
        pred_none = pred_a1[v] == ids.NONE_TOKEN
        present_v = present[v]
        absent_v = ~present_v

        def _rate(mask):
            return float(pred_none[mask].float().mean().item()) if mask.any() else None

        metrics = {
            "n_queries": int(v.sum()),
            "exact_match_present": float(exact_match[present_v].float().mean().item())
            if present_v.any() else 0.0,
            "abstention_accuracy": _rate(absent_v),
            "false_abstention_rate": _rate(present_v),
            "ece_first_answer": kmetrics.ece(conf_key[v], a1_ok, n_bins=15),
        }
        signals_at_key = {"max_prob": conf_key[v], "entropy": entropy_key[v]}
        for name in ("u_tok", "u_span", "jepa_logvar", "jepa_err", "loop_delta_last"):
            if name in aux:
                signals_at_key[name] = aux[name][v]
        return RecallMetrics(
            N=N, n_docs=B, n_queries=metrics["n_queries"],
            exact_match_present=metrics["exact_match_present"],
            abstention_accuracy=metrics["abstention_accuracy"],
            false_abstention_rate=metrics["false_abstention_rate"],
            ece_first_answer=metrics["ece_first_answer"],
            correct_a1=a1_ok,
            exact_match=exact_match,
            signals_at_key=signals_at_key,
            entropy_at_key=entropy_key[v],
        )
