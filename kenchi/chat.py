"""The Kenchi chat runner (src/kenchi/chat.py).

KenchiChat implements the tool loop of docs/CHAT_FORMAT.md against a
pluggable backend, using the canonical renderer (kenchi_sft.chat.render_tokens
via kenchi.tokenizer.encode_conversation) so inference and training formats
cannot drift:

- generate_tokens(model, idx, ...): the model's cached step path; returns the
  generated ids per sequence (from `<|assistant|>` through the first
  `<|end|>` inclusive).
- generate_batch(model, prompts, ...): batched generation. The model's cached
  `generate` needs equal-length prompts, so prompts are bucketed by length and
  each bucket is generated as one batch; results equal single-sequence
  generation (tests/test_chat.py). Returns per-sequence log-probs of the
  generated tokens and the u-head output at the first generated position.
- RemoteSearchBackend(url, token): the HTTP search client (wire shape agreed
  with the other worker; mocked in tests).
- CLI: ``python -m kenchi.chat --weights <file> --config <file> [--index <dir>
  | --search-url <url>] [--no-tools]``.
"""

from __future__ import annotations

import os

import argparse
import contextlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from kenchi.render import (
    DEFAULT_SYSTEM_TEXT,
    NO_TOOL_SYSTEM_TEXT,
    assistant_turn,
    format_open_page_result,
    format_results,
    result_turn,
    system_turn,
    tool_call_turn,
    tool_result_turn,
    user_turn,
)
from kenchi import tokenizer
from kenchi import fastgen
from kenchi import io as kio
from kenchi.model import KenchiConfig
from kenchi.tokenizer import encode_conversation

__all__ = ["ABSTENTION_TEXT", "ConvRecord", "GenResult", "KenchiChat", "RemoteSearchBackend",
           "Trace", "generate_batch", "generate_tokens", "main", "parse_generated",
           "run_conversation", "run_conversations", "search_batch"]


def _autocast_ctx(device, dtype=torch.bfloat16):
    """bf16 autocast on CUDA (the model has no key-value cache; generation is
    forward-only), disabled elsewhere."""
    if isinstance(device, torch.device):
        device = device.type
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()

#: Runner fallback when the model has no answer and no tool calls left.
ABSTENTION_TEXT = "I couldn't find a reliable source for that."

#: Built-in tool name -> backend method (BM25WikiBackend/StaticBackend expose
#: `search`/`open_page`).
_TOOL_METHODS = {"web_search": "search", "open_page": "open_page"}


# ------------------------------------------------------------------ generation

@torch.no_grad()
def generate_tokens(model, idx, max_new_tokens=256, temperature=0.0, top_k=0):
    """Generate with the model's cached step path (`Kenchi.generate` semantics,
    without stopping at `<|end|>`).

    `idx`: (B, T) or (T,) token ids. Returns a list of per-sequence generated
    ids: everything from the assistant turn's first token through the first
    `<|end|>` inclusive (or all `max_new_tokens` when no `<|end|>` is emitted).
    """
    if idx.dim() == 1:
        idx = idx.unsqueeze(0)
    B, T = idx.shape
    n = model._n_loops(None)
    states = model._init_states(B, n)
    logits = None
    ctx = _autocast_ctx(idx.device)
    with ctx:
        for t in range(T):
            logits = model._step_token(idx[:, t], states, n)
        steps = []
        for _ in range(int(max_new_tokens)):
            tok = model._sample(logits, temperature, top_k)
            steps.append([int(x) for x in tok.detach().cpu().tolist()])
            logits = model._step_token(tok, states, n)
    seqs = [[row[b] for row in steps] for b in range(B)]
    return [_cut_at_end(s) for s in seqs]


def _cut_at_end(ids):
    for k, t in enumerate(ids):
        if t == tokenizer.TOKEN_END:
            return ids[:k + 1]
    return ids


def parse_generated(ids) -> Tuple[str, Optional[Tuple[str, Dict[str, Any], Optional[str]]]]:
    """Decode generated ids (from generate_tokens) to ``(text, tool_call)``.

    `text` is the decoded body (after `<|assistant|>`, up to `<|end|>`);
    `tool_call` is None for an answer, or ``(name, arguments, error)`` where
    `error` is set when the JSON of a tool call is invalid (the runner then
    returns a formatted error tool result so the model can retry).
    """
    end = next((k for k, t in enumerate(ids) if t == tokenizer.TOKEN_END), None)
    body = list(ids) if end is None else ids[:end]
    if body and body[0] == tokenizer.TOKEN_ASSISTANT:
        body = body[1:]
    is_tool = bool(body) and body[0] == tokenizer.TOKEN_TOOL_CALL
    payload = body[1:] if is_tool else body
    # anything from another turn marker or the end of the document on is not
    # part of this turn
    stop = {tokenizer.TOKEN_ENDOFTEXT, tokenizer.TOKEN_SYSTEM, tokenizer.TOKEN_USER,
            tokenizer.TOKEN_ASSISTANT, tokenizer.TOKEN_TOOL_RESULT}
    cut = next((k for k, t in enumerate(payload) if t in stop), None)
    if cut is not None:
        payload = payload[:cut]
    text = tokenizer.decode_text(payload)
    if not is_tool:
        return text, None
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(f"tool call JSON is not an object: {text[:80]!r}")
        name = obj.get("name")
        args = obj.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError(f"bad tool-call name: {name!r}")
        if not isinstance(args, dict):
            raise ValueError(f"bad tool-call arguments: {args!r}")
        return text, (name, args, None)
    except (ValueError, TypeError) as e:
        return text, ("", {}, f"invalid tool-call JSON: {e}")


@dataclass
class GenResult:
    """One sequence of generate_batch."""
    text: str
    token_ids: List[int]
    logprobs: List[float]     # log-prob of each generated token (after `<|assistant|>`)
    u_first: Optional[float]   # u-head output at the first generated position


@torch.no_grad()
def _generate_bucket(model, idx, *, max_new_tokens=256, temperature=0.0, top_k=0):
    """generate_tokens-style generation for one equal-length bucket, also
    capturing per-token log-probs and the u-head output at the first
    generated position."""
    B, T = idx.shape
    n = model._n_loops(None)
    states = model._init_states(B, n)
    logits = None
    ctx = _autocast_ctx(idx.device)
    with ctx:
        for t in range(T):
            logits = model._step_token(idx[:, t], states, n)
        steps = []
        for _ in range(int(max_new_tokens)):
            tok = model._sample(logits, temperature, top_k)
            lp = F.log_softmax(logits.float(), dim=-1).gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            steps.append((tok.detach().cpu(), lp.detach().cpu()))
            logits = model._step_token(tok, states, n)
        gen = torch.stack([s[0] for s in steps], dim=1) if steps else idx.new_zeros((B, 0))
        out = model(torch.cat([idx, gen], dim=1), return_aux=True)
    u_all = out.get("u_tok_logit")
    u_first = [None] * B if u_all is None else u_all[:, T - 1].detach().cpu().tolist()
    results = []
    for b in range(B):
        ids = [int(s[0][b]) for s in steps]
        lps = [float(s[1][b]) for s in steps]
        end = next((k for k, t in enumerate(ids) if t == tokenizer.TOKEN_END), None)
        if end is not None:
            ids, lps = ids[:end + 1], lps[:end + 1]
        results.append(GenResult(text=tokenizer.decode_text(ids), token_ids=ids,
                                 logprobs=lps, u_first=u_first[b]))
    return results


@torch.no_grad()
def generate_ragged(model, prompts, *, max_new_tokens=256, temperature=0.0, top_k=0):
    """Generate for prompts of ANY lengths in one batch.

    Uses `kenchi.fastgen` (one parallel pass over the prompts, then cached steps
    for the new tokens only) when the model has the released layout; otherwise
    the slow step path below.  Both give the same tokens.
    """
    if not prompts:
        return []
    if fastgen.supported(model) and not os.environ.get("KENCHI_SLOW_GEN"):
        dev = next(model.parameters()).device
        END = int(tokenizer.TOKEN_END)
        with _autocast_ctx(dev):
            ids, lps = fastgen.generate_ids(model, prompts, max_new_tokens=max_new_tokens,
                                            end_id=END, temperature=temperature, top_k=top_k)
        return [GenResult(text=tokenizer.decode_text(i), token_ids=i, logprobs=l, u_first=None)
                for i, l in zip(ids, lps)]
    return _generate_ragged_steps(model, prompts, max_new_tokens=max_new_tokens,
                                  temperature=temperature, top_k=top_k)


@torch.no_grad()
def _generate_ragged_steps(model, prompts, *, max_new_tokens=256, temperature=0.0, top_k=0):
    """Step-path version of `generate_ragged` (slow: the prompt is fed one token per step).

    The model's cached step path consumes one token per row per step.  Every
    row advances in lockstep: while a row is still inside its prompt it is fed
    its next prompt token; after that it is fed the token it just produced.
    Rows finish at `<|end|>` or after ``max_new_tokens``; the loop ends when
    all rows are finished.  Rows are independent, so the result of each row
    equals single-sequence generation (greedy; tested in tests/test_chat.py).
    ``prompts`` is a list of token-id lists.  Returns GenResult per prompt.
    """
    if not prompts:
        return []
    dev = next(model.parameters()).device
    B = len(prompts)
    lens = [len(p) for p in prompts]
    L = torch.tensor(lens, dtype=torch.long, device=dev)
    maxL = int(max(lens))
    END = int(tokenizer.TOKEN_END)
    P = torch.full((B, maxL), END, dtype=torch.long, device=dev)
    for i, p_ in enumerate(prompts):
        P[i, :len(p_)] = torch.tensor(list(p_), dtype=torch.long, device=dev)
    n = model._n_loops(None)
    states = model._init_states(B, n)
    last = P[:, 0].clone()
    M = int(max_new_tokens)
    gen = torch.full((B, M), END, dtype=torch.long, device=dev)
    glp = torch.zeros((B, M), dtype=torch.float32, device=dev)
    count = torch.zeros(B, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    rows_all = torch.arange(B, device=dev)
    with _autocast_ctx(dev):
        for t in range(maxL + M):
            in_prompt = t < L
            tok_in = torch.where(in_prompt, P[:, min(t, maxL - 1)], last)
            logits = model._step_token(tok_in, states, n)
            rec = (t + 1 >= L) & ~done            # rows whose next token is generated now
            if bool(rec.any()):
                tok = model._sample(logits, temperature, top_k)
                lp = F.log_softmax(logits.float(), dim=-1).gather(-1, tok.unsqueeze(-1)).squeeze(-1)
                rows = rows_all[rec]
                gen[rows, count[rows]] = tok[rows]
                glp[rows, count[rows]] = lp[rows]
                count[rows] += 1
                last = torch.where(rec, tok, last)
                done = done | (rec & (tok == END)) | (count >= M)
            if bool(done.all()):
                break
    gen_c, glp_c, count_c = gen.cpu().tolist(), glp.cpu().tolist(), count.cpu().tolist()
    results = []
    for i in range(B):
        ids = gen_c[i][:count_c[i]]
        lps = glp_c[i][:count_c[i]]
        results.append(GenResult(text=tokenizer.decode_text(ids), token_ids=ids,
                                 logprobs=lps, u_first=None))
    return results


#: rows per generation batch (memory of the step path grows with rows x length)
GEN_MAX_BATCH = int(os.environ.get("KENCHI_GEN_BATCH") or 128)


def generate_batch(model, prompts, *, max_new_tokens=256, temperature=0.0, top_k=0):
    """Batched generation for evaluation and calibration: prompts are sorted by
    length and generated in chunks of GEN_MAX_BATCH rows with `generate_ragged`
    (one batch per chunk whatever the lengths).  Returns GenResult in input order."""
    if isinstance(prompts, str):
        prompts = [prompts]
    ids = []
    for p_ in prompts:
        if isinstance(p_, str):
            p_ = tokenizer.encode_text(p_)
        ids.append([int(t) for t in p_])
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    results: List[Optional[GenResult]] = [None] * len(ids)
    for k in range(0, len(order), GEN_MAX_BATCH):
        chunk = order[k:k + GEN_MAX_BATCH]
        rs = generate_ragged(model, [ids[i] for i in chunk], max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k)
        for i, r in zip(chunk, rs):
            results[i] = r
    return results  # type: ignore[return-value]


@dataclass
class ConvRecord:
    """One conversation of `run_conversation` / `run_conversations`."""
    turns: List[Dict[str, Any]]
    final_text: str
    malformed_tool_call: bool
    runaway: bool
    tool_calls_used: int
    searches: int = 0
    open_pages: int = 0


def search_batch(backend, queries, k=5):
    """One batched search call for several queries: returns one result list
    per query -- the backend's `search_batch` when it has one (the remote
    client being written by another worker), else a loop over `search`."""
    queries = [str(q) for q in queries]
    if backend is None:
        return [[] for _ in queries]
    sb = getattr(backend, "search_batch", None)
    if callable(sb):
        out = sb(queries, k=k)
        out = list(out or [])
        if len(out) != len(queries):
            return [list(backend.search(q, k=k)) for q in queries]
        return [list(r or []) for r in out]
    return [list(backend.search(q, k=k)) for q in queries]


class _Live:
    """One conversation being advanced by the batched runner."""

    def __init__(self, messages, system):
        self.turns = [dict(t) for t in messages]
        self.system = system
        self.offset = 0
        self.pending = []              # tool calls queued by the latest generation
        self.tool_calls_used = 0
        self.searches = 0
        self.open_pages = 0
        self.malformed = False
        self.runaway = False
        self.done = False
        self.final_text = ""
        self.answer_mean_logprob = None

    def record(self) -> ConvRecord:
        return ConvRecord(turns=self.turns, final_text=self.final_text,
                          malformed_tool_call=self.malformed, runaway=self.runaway,
                          tool_calls_used=self.tool_calls_used, searches=self.searches,
                          open_pages=self.open_pages)


def _fit_context(turns, system, limit):
    """Token ids of the conversation, at most ``limit`` long: the oldest tool
    results are dropped first, then the oldest other turns (as KenchiChat does)."""
    # The prompt ends with <|assistant|> so the model has to write a turn; left
    # to choose, it sometimes ended the document instead (seen 2026-11-19).
    limit = int(limit) - 1
    while True:
        idx = encode_conversation(turns, system)
        if len(idx) <= limit:
            return idx + [tokenizer.TOKEN_ASSISTANT]
        tool_results = [k for k, t in enumerate(turns) if t.get("role") == "tool_result"]
        if tool_results:
            turns.pop(tool_results[0])
            continue
        others = [k for k, t in enumerate(turns)
                  if t.get("role") in ("user", "assistant", "tool_call")]
        if len(others) > 1:
            turns.pop(others[0])
            continue
        return idx[-limit:] + [tokenizer.TOKEN_ASSISTANT]


def _run_rounds(lives, model, backend, *, max_searches=2, max_new_tokens=256,
                temperature=0.0) -> List[ConvRecord]:
    """Advance every live conversation to a final answer (or to the search
    budget): each generation step produces one turn per conversation; every
    pending search of a step goes into ONE `search_batch` call."""
    guard = 0
    while True:
        progress = False
        # -- generation step: one turn per live conversation without pending searches
        targets = [lv for lv in lives if not lv.done and not lv.pending]
        if targets:
            limit = max(64, int(getattr(model.cfg, "seq_len", 2048)) - int(max_new_tokens) - 1)
            gens = generate_batch(model,
                                  [_fit_context(lv.turns, lv.system, limit) for lv in targets],
                                  max_new_tokens=max_new_tokens, temperature=temperature)
            for lv, gen in zip(targets, gens):
                progress = True
                lv.rounds = getattr(lv, "rounds", 0) + 1
                if lv.rounds > max_searches + 2:
                    # calls to tools that are not there, or malformed calls, over and
                    # over: stop (one such loop ran 28 rounds in an evaluation)
                    lv.runaway = True
                    lv.final_text = ABSTENTION_TEXT
                    lv.turns.append(assistant_turn(ABSTENTION_TEXT))
                    lv.done = True
                    continue
                text, tc = parse_generated(gen.token_ids)
                if tc is None:
                    lv.turns.append(assistant_turn(text))
                    lv.final_text = text
                    if gen.logprobs:
                        lv.answer_mean_logprob = float(sum(gen.logprobs) / len(gen.logprobs))
                    lv.done = True
                    continue
                name, args, err = tc
                if err is not None or not name or not isinstance(args, dict):
                    # a malformed tool call is recorded; retry once per budget
                    lv.malformed = True
                    lv.turns.append(result_turn(
                        f"Error: {err or 'invalid tool call'}. Please retry with valid JSON."))
                    continue
                lv.tool_calls_used += 1
                lv.turns.append(tool_call_turn(name, args))
                if backend is None or name not in ("web_search", "open_page"):
                    lv.turns.append(result_turn(f"Error: tool {name!r} is not available."))
                    continue
                if lv.searches >= max_searches:
                    lv.runaway = True
                    lv.final_text = ABSTENTION_TEXT
                    lv.turns.append(assistant_turn(ABSTENTION_TEXT))
                    lv.done = True
                    continue
                if name == "web_search":
                    lv.pending.append(("web_search", str(args.get("query", ""))))
                else:
                    lv.pending.append(("open_page", str(args.get("url", ""))))
        # -- search step: every pending search of every conversation in ONE call
        pend = []
        for lv in lives:
            if not lv.done:
                for kind, q in lv.pending:
                    pend.append((lv, kind, q))
                    if kind == "web_search":
                        lv.searches += 1
                    else:
                        lv.open_pages += 1
                lv.pending = []
        if pend:
            progress = True
            try:
                results = search_batch(backend, [q for _, _, q in pend])
            except Exception:  # noqa: BLE001 -- a failing tool is a result, not a crash
                results = [None] * len(pend)
            for (lv, kind, q), res in zip(pend, results):
                if res is None:
                    name = "web_search" if kind == "web_search" else "open_page"
                    lv.turns.append(result_turn(f"Error: tool {name!r} failed."))
                elif kind == "web_search":
                    res = res if isinstance(res, list) else []
                    lv.turns.append(result_turn(format_results(res, offset=lv.offset)))
                    lv.offset += len(res)
                else:
                    page = backend.open_page(q) if backend is not None else {}
                    lv.turns.append(result_turn(format_open_page_result(
                        page.get("title", ""), page.get("url", q), page.get("text", ""))))
                    lv.offset += 1
        if all(lv.done for lv in lives):
            break
        guard += 1
        if not progress or guard > 4 * (max_searches + 2) + 8:
            for lv in lives:
                if not lv.done:
                    lv.runaway = True
                    lv.final_text = ABSTENTION_TEXT
                    lv.turns.append(assistant_turn(ABSTENTION_TEXT))
                    lv.done = True
            break
    return [lv.record() for lv in lives]


def run_conversations(model, tok, messages_list, search_backend=None, max_searches=2,
                      max_new_tokens=256, temperature=0.0, system=None):
    """Batched chat loop (the loop `KenchiChat.reply` runs per conversation):
    every pending search of a step goes into ONE `search_batch` call. Returns
    a list of `(final_text, ConvRecord)`, one per input conversation, in
    input order. `tok` is the tokenizer used for rendering (the module-level
    `kenchi.tokenizer` is used underneath)."""
    system = system if system is not None else DEFAULT_SYSTEM_TEXT
    lives = [_Live(messages, system) for messages in messages_list]
    records = _run_rounds(lives, model, search_backend, max_searches=max_searches,
                          max_new_tokens=max_new_tokens, temperature=temperature)
    return [(lv.final_text, rec) for lv, rec in zip(lives, records)]


def run_conversation(model, tok, messages, search_backend=None, max_searches=2,
                     max_new_tokens=256, temperature=0.0, system=None):
    """One conversation through the chat loop of docs/CHAT_FORMAT.md:
    `(final_text, record)` (see `run_conversations`)."""
    return run_conversations(model, tok, [messages], search_backend=search_backend,
                             max_searches=max_searches, max_new_tokens=max_new_tokens,
                             temperature=temperature, system=system)[0]


# ------------------------------------------------------------------ runner

@dataclass
class Trace:
    """What KenchiChat.reply returned, for calibration and evaluation."""
    turns: List[Dict[str, Any]]
    final_text: str
    malformed_tool_call: bool = False
    tool_calls_used: int = 0
    runaway: bool = False
    u_first: Optional[float] = None
    answer_mean_logprob: Optional[float] = None
    abstained: bool = False


class KenchiChat:
    """The chat runner: the tool loop of docs/CHAT_FORMAT.md.

    `reply(messages) -> (text, trace)`: generate until `<|end|>`; if the turn
    is a tool call, parse the JSON strictly (on invalid JSON append a
    formatted error tool result so the model can retry once), call the
    pluggable backend, render the result turn exactly as the training data
    does (kenchi_sft renderers) and continue; stop after `max_tool_calls`
    with an abstention if no answer. Context is managed to
    `max_context_tokens` (the oldest tool results are dropped first).
    """

    def __init__(self, model, backend=None, system: Optional[str] = None,
                 max_tool_calls=3, max_context_tokens=2048, max_new_tokens=256,
                 temperature=0.0, top_k=0, abstain_threshold: Optional[float] = None):
        self.model = model
        model.eval()
        self.backend = backend
        self.system = system if system is not None else DEFAULT_SYSTEM_TEXT
        self.max_tool_calls = int(max_tool_calls)
        self.max_context_tokens = int(max_context_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        #: Replaces answers whose u-head output (confidence = sigmoid(u_first))
        #: is below the threshold with the abstention text.
        self.abstain_threshold = abstain_threshold
        #: Cumulative count of rendered results so far in the conversation
        #: (docs/CHAT_FORMAT.md: result numbers run cumulatively).
        self._result_offset = 0

    # ------------------------------------------------------------------ loop
    def _generate(self, idx):
        return generate_tokens(self.model, idx, max_new_tokens=self.max_new_tokens,
                               temperature=self.temperature, top_k=self.top_k)

    def reply(self, messages) -> Tuple[str, Trace]:
        turns = [dict(t) for t in messages]
        malformed = 0
        tool_calls_used = 0
        for _ in range(self.max_tool_calls):
            idx = self._context_idx(turns)
            gen = self._generate(idx)[0]
            text, tc = parse_generated(gen)
            if tc is not None:
                name, args, err = tc
                tool_calls_used += 1
                if err is not None or not name:
                    malformed += 1
                    turns.append(result_turn(
                        f"Error: {err or 'invalid tool call'}. Please retry with valid JSON."))
                    continue
                turns.append(tool_call_turn(name, args))
                turns.append(result_turn(self._run_tool(name, args)))
                continue
            turns.append(assistant_turn(text))
            trace = Trace(turns=turns, final_text=text,
                          malformed_tool_call=malformed > 0,
                          tool_calls_used=tool_calls_used, runaway=False)
            trace.u_first, trace.answer_mean_logprob = self.analyze(turns)
            self._apply_abstain_threshold(trace)
            return trace.final_text, trace
        abstention = ABSTENTION_TEXT
        turns.append(assistant_turn(abstention))
        trace = Trace(turns=turns, final_text=abstention,
                      malformed_tool_call=malformed > 0,
                      tool_calls_used=tool_calls_used, runaway=True)
        trace.u_first, trace.answer_mean_logprob = self.analyze(turns)
        self._apply_abstain_threshold(trace)
        return trace.final_text, trace

    def reply_many(self, messages_list, chunk: int = 512):
        """Batched ``reply``: many conversations advance together (one
        generation batch per step, one search call per step).  Returns
        ``[(text, Trace), ...]`` in input order.  No u-head analysis and no
        abstain threshold (use ``reply`` for those)."""
        out = []
        for i in range(0, len(messages_list), int(chunk)):
            lives = [_Live(m, self.system) for m in messages_list[i:i + int(chunk)]]
            _run_rounds(lives, self.model, self.backend, max_searches=self.max_tool_calls,
                        max_new_tokens=self.max_new_tokens, temperature=self.temperature)
            for lv in lives:
                tr = Trace(turns=lv.turns, final_text=lv.final_text,
                           malformed_tool_call=lv.malformed, tool_calls_used=lv.tool_calls_used,
                           runaway=lv.runaway, u_first=None,
                           answer_mean_logprob=lv.answer_mean_logprob)
                out.append((lv.final_text, tr))
        return out

    # ------------------------------------------------------------------ context
    def _context_idx(self, turns) -> torch.Tensor:
        while True:
            idx = encode_conversation(turns, self.system)
            if len(idx) <= self.max_context_tokens - 1:
                return torch.tensor([idx + [tokenizer.TOKEN_ASSISTANT]], dtype=torch.long)
            tool_results = [k for k, t in enumerate(turns) if t.get("role") == "tool_result"]
            if tool_results:
                turns.pop(tool_results[0])
                continue
            others = [k for k, t in enumerate(turns)
                      if t.get("role") in ("user", "assistant", "tool_call")]
            if others:
                turns.pop(others[0])
                continue
            return torch.tensor([idx[:self.max_context_tokens - 1] + [tokenizer.TOKEN_ASSISTANT]],
                                dtype=torch.long)

    # ------------------------------------------------------------------ tools
    def _find_tool(self, name) -> Optional[Callable]:
        backend = self.backend
        if backend is None:
            return None
        method = _TOOL_METHODS.get(name, name)
        if isinstance(backend, Mapping):
            fn = backend.get(name)
            if not callable(fn):
                fn = backend.get(method)
        else:
            fn = getattr(backend, name, None)
            if not callable(fn):
                fn = getattr(backend, method, None)
        return fn if callable(fn) else None

    def _run_tool(self, name, args) -> str:
        fn = self._find_tool(name)
        if fn is None:
            return f"Error: tool {name!r} is not available."
        try:
            result = fn(**args)
        except Exception as e:  # noqa: BLE001 -- a failing tool is a result
            return f"Error: tool {name!r} failed."      # one fixed string; it is in the fine-tuning set
        text = self._format_result(name, result, offset=self._result_offset)
        try:
            if name == "web_search" and isinstance(result, list):
                self._result_offset += len(result)
            elif name == "open_page":
                # An opened page gets the next result number (W15b defect 1).
                self._result_offset += 1
        except Exception:  # noqa: BLE001
            pass
        return text

    @staticmethod
    def _format_result(name, result, offset=0) -> str:
        # Reuse the training-data renderers (format_results /
        # format_open_page_result) so the tool-result turns the model saw in
        # training and the ones it sees now cannot drift; result numbers run
        # cumulatively through the conversation (W15b defect 1).
        if name == "web_search":
            return format_results(result if isinstance(result, list) else [], offset=offset)
        if name == "open_page":
            if isinstance(result, dict):
                return format_open_page_result(result.get("title", ""),
                                               result.get("url", ""),
                                               result.get("text", ""))
            return str(result)
        return result if isinstance(result, str) else json.dumps(result, default=str)

    # ------------------------------------------------------------------ analysis
    def analyze(self, turns) -> Tuple[Optional[float], Optional[float]]:
        """``(u_first, answer_mean_logprob)`` of the final assistant turn.

        ``u_first`` is the u-head output at the final assistant turn's
        `<|assistant|>` position -- the position whose output scores the first
        generated token; ``answer_mean_logprob`` is the mean log-prob of the
        final-answer tokens.
        """
        idx = encode_conversation(turns, self.system)
        if not idx:
            return None, None
        model = self.model
        was = model.training
        model.eval()
        try:
            with torch.no_grad():
                out = model(torch.tensor([idx], dtype=torch.long), return_aux=True)
                u = out.get("u_tok_logit")
                logits = out.get("logits")
                if logits is None:
                    return None, None
                lp = F.log_softmax(logits[0].float(), dim=-1)
            pos = [i for i, t in enumerate(idx) if t == tokenizer.TOKEN_ASSISTANT]
            if not pos:
                return None, None
            p = pos[-1]
            q = next((i for i in range(p + 1, len(idx)) if idx[i] == tokenizer.TOKEN_END),
                     len(idx))
            if q <= p + 1:
                return None, None
            toks = torch.tensor(idx[p + 1:q], dtype=torch.long)
            positions = torch.arange(p + 1, q) - 1
            answer_lps = lp[positions].gather(-1, toks.unsqueeze(-1)).squeeze(-1)
            u_first = None if u is None else float(u[0][p])
            return u_first, float(answer_lps.mean())
        finally:
            if was:
                model.train()

    def _apply_abstain_threshold(self, trace: Trace) -> None:
        if self.abstain_threshold is None or trace.u_first is None:
            return
        conf = float(torch.sigmoid(torch.tensor(trace.u_first, dtype=torch.float32)))
        if conf < float(self.abstain_threshold):
            trace.abstained = True
            trace.final_text = ABSTENTION_TEXT
            if trace.turns:
                trace.turns[-1] = assistant_turn(ABSTENTION_TEXT)


# ------------------------------------------------------------------ remote search client

class RemoteSearchBackend:
    """HTTP client for the search service written by another worker.

    Wire shape (agreed; mocked in tests/test_chat.py):

    - POST ``{"op": "search"|"open_page", "queries": [...]}`` to the service URL
      -> JSON ``{"results": [...]}``: for ``search``, one list of result dicts
      ``{"title", "url", "snippet"}``; for ``open_page``, the page dict
      ``{"title", "url", "text", "found"}``. Optional
      ``Authorization: Bearer <token>`` header when `token` is given.
    """

    def __init__(self, url, token=None, timeout=15.0):
        self.url = str(url)
        self.token = token
        self.timeout = float(timeout)

    def _post(self, op, queries):
        body = json.dumps({"op": op, "queries": list(queries)}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def search(self, query, k=5):
        payload = self._post("search", [query])
        results = payload.get("results")
        if not isinstance(results, list):
            return []
        return [r for r in results if isinstance(r, dict)][: int(k)]

    def search_batch(self, queries, k=5):
        """One call per query (the remote service implements native batching
        in the client `kenchi_search.RemoteSearchBackend`)."""
        return [self.search(q, k=k) for q in queries]

    def open_page(self, url, max_words=600):
        payload = self._post("open_page", [url])
        results = payload.get("results")
        page = results[0] if isinstance(results, list) and results else None
        if not isinstance(page, dict):
            return {"title": Path(url).name, "url": url, "text": "", "found": False}
        return page


# ------------------------------------------------------------------ CLI

def _model_config_from(path) -> KenchiConfig:
    raw = json.loads(Path(path).expanduser().read_text())
    if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
        return KenchiConfig.from_dict(raw["model"])
    return KenchiConfig.from_dict(raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m kenchi.chat",
        description="Kenchi chat runner (docs/CHAT_FORMAT.md).")
    ap.add_argument("--weights", required=True,
                    help="delivered weights file (safetensors, or the torch.save resume/final ckpt)")
    ap.add_argument("--config", required=True,
                    help="model config JSON (e.g. ~/kenchi_ckpt/*_final_config.json)")
    ap.add_argument("--index", default=None, help="BM25 index dir (kenchi_search)")
    ap.add_argument("--search-url", default=None, help="remote search service URL")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--max-tool-calls", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--question", default=None)
    args = ap.parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, mcfg = kio.load_model(args.weights, args.config, device="cpu")
    assert mcfg.seq_len > 0

    backend = None
    system = DEFAULT_SYSTEM_TEXT
    if not args.no_tools:
        if args.search_url:
            backend = RemoteSearchBackend(args.search_url)
        else:
            index = args.index
            if index is None:
                default_index = Path("~/kenchi_data/search/index_psg").expanduser()
                if default_index.is_dir():
                    index = default_index
            if index:
                from kenchi_search.bm25 import BM25WikiBackend
                backend = BM25WikiBackend(index)
    else:
        system = NO_TOOL_SYSTEM_TEXT

    chat = KenchiChat(model, backend=backend, system=system,
                      max_tool_calls=args.max_tool_calls,
                      max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature)
    q = args.question
    if q is None:
        q = input("question: ")
    text, trace = chat.reply([system_turn(system), user_turn(q)])
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
