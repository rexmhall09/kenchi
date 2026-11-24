"""One model turn on OpenAI-style messages, the tool loop, and the self-check.

This is what the PyTorch runner (run.py) and the server (serve.py) share.  The MLX runner has the
same logic on its own arrays (run_mlx.py).

    a = Assistant(model, meta)                       # a PyTorch model (or an engine); meta: the weights file's metadata
    msg = a.complete(messages, tools)                # -> {"role": "assistant", "content"/"tool_calls", ...}
    text, trace = a.ask(messages, search=fn)         # runs the search calls itself

History.  `Assistant(history=...)` says what the model is shown of a conversation:
"last" (the default): only the last user message and what followed it; "compact": the earlier turns
as plain exchanges (user text + final answer, see convo.compact_history), at most 400 tokens of
them; "all": every message as it was sent.  The environment variable KENCHI_HISTORY sets the default.

Evidence line.  After a search the model may first write one line that says what it relies on
(`[2] says: "..."` or `No result states this.`, see convo.py).  `complete()` takes the line off
the reply (it is returned as `reasoning_content`) and checks it against the tool result: a quote
that is not in the result it names, or "No result states this." followed by an answer anyway,
replaces the answer by "I could not find a reliable source for that."  `evidence_check=False`
turns the replacement off.

The self-check is a small head on the model's last layer (`u_head`, 2 x Linear, two outputs).  It
was fitted, with the rest of the model frozen, on the model's own replies to held-out prompts: does
this reply hold up?  Output 0 scores an answer written from tool results; under its threshold
(stored in the weights file) the answer is replaced by "I could not find a reliable source for
that."  Output 1 scores a reply to a request without any tools, where the model answers from
memory; under its threshold the reply becomes "I don't know."  `self_check=False` turns both off.
Tool calls are never touched.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import convo, format_v2
from .abstain import looks_abstained

ABSTAIN_TEXT = "I could not find a reliable source for that."
DONT_KNOW_TEXT = "I don't know."
CONTEXT = 2048
HISTORY_MODES = ("last", "compact", "all")
COMPACT_BUDGET = 400            # tokens of earlier exchanges shown in "compact" mode


def read_metadata(weights) -> Dict[str, str]:
    """The string metadata of a safetensors file (the self-check threshold `u_threshold` lives there)."""
    import struct
    from pathlib import Path

    with open(Path(weights).expanduser(), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
    return dict(header.get("__metadata__") or {})


def read_threshold(meta: Optional[Dict[str, str]], key: str = "u_threshold") -> Optional[float]:
    """A self-check threshold stored in the weights file's metadata, or None (not fitted / not used)."""
    try:
        return float((meta or {}).get(key))
    except (TypeError, ValueError):
        return None


def results_text(results: Sequence[Dict[str, Any]], k: int = 5, words: int = 60) -> str:
    """Search results as the model saw them most often in training: `[n] Title - URL`, then the snippet."""
    rs = list(results)[:k]
    if not rs:
        return "No results."
    return "\n".join(f"[{i}] {r.get('title', '')} - {r.get('url', '')}\n{' '.join(str(r.get('snippet', '')).split()[:words])}"
                     for i, r in enumerate(rs, 1))


def fit_history(messages: List[Dict[str, Any]], tools, room: int, count: Callable[[List[Dict[str, Any]], Any], int]):
    """Drop the oldest exchanges until the prompt leaves `room` tokens for the answer.

    An exchange is a user message and everything up to the next user message; a leading system
    message is kept.  Returns the messages to send (the last exchange is never dropped)."""
    msgs = list(messages)
    while count(msgs, tools) > CONTEXT - room:
        start = 1 if msgs and msgs[0].get("role") == "system" else 0
        users = [i for i in range(start, len(msgs)) if msgs[i].get("role") == "user"]
        if len(users) < 2:
            break
        del msgs[users[0]:users[1]]
    return msgs


def last_exchange(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The question being asked now, without the conversation before it: the system message(s) at
    the top, then everything from the last user message on (its tool calls and tool results).

    Why: this model cannot use earlier turns.  After a single "hi" it stops searching and answers
    the next question with a line about itself (measured: tests/defects/history.py).  So by
    default every runner shows it one question at a time."""
    msgs = list(messages)
    last = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=None)
    if last is None:
        return msgs
    n = 0
    while n < last and msgs[n].get("role") in ("system", "developer"):
        n += 1
    return msgs[:n] + msgs[last:]


def default_history(config_path) -> Optional[str]:
    """The history mode a model's config.json asks for (`"history_default"`), or None.  The runners use
    it when no --history is given: weights that can use a conversation say "compact" there."""
    try:
        from pathlib import Path as _Path
        value = json.loads(_Path(config_path).expanduser().read_text()).get("history_default")
        return str(value) if value else None
    except Exception:  # noqa: BLE001
        return None


def history_mode(value=None) -> str:
    """"last", "compact" or "all" for what a caller passed: one of those names, True (= "all"),
    False (= "last"), or None: the environment variable KENCHI_HISTORY, else "last"."""
    if value is None:
        value = os.environ.get("KENCHI_HISTORY") or "last"
    if value is True:
        return "all"
    if value is False:
        return "last"
    mode = str(value).strip().lower()
    if mode not in HISTORY_MODES:
        raise ValueError(f"history must be one of {', '.join(HISTORY_MODES)} (got {value!r})")
    return mode


_HEADER_TOKENS: Optional[int] = None


def turn_tokens(messages: List[Dict[str, Any]]) -> int:
    """The tokens of these messages rendered with format_v2 (the turns only: no system block, no
    generation prompt).  This is the `count` of convo.compact_history."""
    global _HEADER_TOKENS
    if _HEADER_TOKENS is None:
        _HEADER_TOKENS = len(format_v2.render([], None, add_generation_prompt=False))
    return len(format_v2.render(messages, None, add_generation_prompt=False)) - _HEADER_TOKENS


def shown_messages(messages: List[Dict[str, Any]], mode: str = "last", tools=None, room: int = 256,
                   count: Optional[Callable[[List[Dict[str, Any]], Any], int]] = None) -> List[Dict[str, Any]]:
    """The messages the model is shown in a history mode (see history_mode).

    "last": last_exchange.  "compact": convo.compact_history with at most COMPACT_BUDGET tokens of
    earlier exchanges; if the prompt still does not leave `room` tokens for the answer, more of the
    earlier exchanges are dropped, oldest first, never the last one.  "all": the messages as they are.
    Applying it twice gives the same messages."""
    if mode == "last":
        return last_exchange(messages)
    if mode == "compact":
        msgs = convo.compact_history(messages, count=turn_tokens, budget=COMPACT_BUDGET)
        return fit_history(msgs, tools, room, count or (lambda m, t: len(format_v2.render(m, t))))
    return list(messages)


class TorchEngine:
    """The PyTorch model behind an Assistant.  (run_mlx.py has the MLX one, with the same two methods.)"""

    def __init__(self, model):
        self.model = model.eval()
        self.has_head = getattr(model, "u_head", None) is not None

    prefer_first_ok = True      # generate() takes prefer_first (see fastgen.generate_ids)

    def generate(self, prompt: List[int], max_new_tokens: int, temperature: float = 0.0, ban_first=None,
                 prefer_first=None):
        """-> (new token ids up to and with <|end|>, the last-layer features the last one was predicted
        from, the mean and the lowest log-probability of the new tokens)."""
        import torch

        from . import fastgen
        with torch.no_grad():
            ids, lps, _pre, last = fastgen.generate_ids(self.model, [prompt], max_new_tokens=max_new_tokens,
                                                        end_id=format_v2.END, temperature=temperature,
                                                        return_features=True, ban_first=ban_first,
                                                        prefer_first=prefer_first)
        return ids[0], last, (sum(lps[0]) / len(lps[0]) if lps[0] else 0.0), (min(lps[0]) if lps[0] else 0.0)

    def check(self, features, channel: int = 0) -> float:
        """The self-check head on those features: the logit of "this reply holds up".
        Channel 0: an answer written after a tool result.  Channel 1: a reply without any tools."""
        import torch

        with torch.no_grad():
            dev = next(self.model.parameters()).device
            return float(self.model.u_head(features.to(dev).float())[0, channel])


class Assistant:
    def __init__(self, engine, meta: Optional[Dict[str, str]] = None, self_check: bool = True,
                 max_new_tokens: int = 256, history=None, evidence_check: bool = True):
        #: "last" (default; also False): the model sees only the last user message and what followed
        #: it (see last_exchange).  "compact": the earlier turns as plain exchanges (see
        #: shown_messages).  "all" (also True): the whole conversation as it was sent; it answers
        #: worse.  None: the environment variable KENCHI_HISTORY, else "last".
        self.history = history_mode(history)
        #: False: an evidence line that does not hold no longer replaces the answer (the line is
        #: still taken off the reply and returned)
        self.evidence_check = bool(evidence_check)
        #: after a tool result the reply opens with an evidence quote unless "No result states this."
        #: is more than this many times as likely as all "[n] says" openings together (see complete)
        self.evidence_ratio = read_threshold(meta, "evidence_ratio") or 1.0
        #: True only for weights trained to open an answer with the evidence line (metadata
        #: `evidence_line` = "1", or KENCHI_EVIDENCE_LINE=1); other weights are left alone
        import os as _os
        self.evidence_line = str((meta or {}).get("evidence_line", _os.environ.get("KENCHI_EVIDENCE_LINE", "0"))) == "1"
        if not hasattr(engine, "check"):                       # a bare PyTorch model
            engine = TorchEngine(engine)
        self.engine = engine
        self.threshold = read_threshold(meta)
        self.threshold_notools = read_threshold(meta, "u_threshold_notools")
        self.lp_weight = ((read_threshold(meta, "u_lp_weight") or 0.0, read_threshold(meta, "u_lpmin_weight") or 0.0),
                          (read_threshold(meta, "u_lp_weight_notools") or 0.0,
                           read_threshold(meta, "u_lpmin_weight_notools") or 0.0))
        self.self_check = bool(self_check) and self.threshold is not None and bool(getattr(engine, "has_head", False))
        self.max_new_tokens = int(max_new_tokens)

    @staticmethod
    def _no_ids() -> List[int]:
        from . import tokenizer as tk
        return [tk.encode_text(convo.NO_EVIDENCE)[0]]

    @staticmethod
    def _cite_ids() -> List[int]:
        from . import tokenizer as tk
        return sorted({tk.encode_text(f"[{n}] says")[0] for n in range(1, 11)})

    # ---------------------------------------------------------------- one turn
    def count(self, messages, tools=None) -> int:
        return len(format_v2.render(messages, tools))

    def complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
                 max_new_tokens: Optional[int] = None, temperature: float = 0.0) -> Dict[str, Any]:
        """One assistant message for these messages (an OpenAI-style dict; see format_v2.parse_output).

        Extra keys: `prompt_tokens`, `completion_tokens`, and `self_check` when the head scored the
        answer: {"margin" (score minus threshold; kept when >= 0), "passed", "kind", "replaced": the
        model's own text when it did not pass}.  When an answer after a tool result opened with an
        evidence line: `reasoning_content` (the line; `content` is the reply without it) and
        `evidence`: {"ok", "line", "replaced": the model's answer when the line did not hold and
        the answer was replaced}."""
        n = int(max_new_tokens or self.max_new_tokens)
        messages = shown_messages(messages, self.history, tools, n, self.count)
        prompt = format_v2.render(messages, tools)
        if len(prompt) + n > CONTEXT:
            raise ValueError(f"the prompt is {len(prompt)} tokens; with {n} new tokens it does not fit the "
                             f"{CONTEXT}-token context")
        # no tools in the request: the reply may not open a tool call (there is nothing to call)
        after_tool = bool(messages) and messages[-1].get("role") in ("tool", "tool_result")
        extra = {}
        if after_tool and self.evidence_line and getattr(self.engine, "prefer_first_ok", False):
            # After a tool result the reply opens with the evidence line: "No result states this." or
            # "[n] says: ...".  The second is split over the result numbers, so a single "No" can win
            # although the model thinks an answer is more likely.  Compare the totals instead.
            extra["prefer_first"] = (self._no_ids(), self._cite_ids(), self.evidence_ratio)
        ids, last, mean_lp, min_lp = self.engine.generate(prompt, n, temperature,
                                                          ban_first=None if tools else [format_v2.TOOL_CALL], **extra)
        msg = format_v2.parse_output(ids)
        msg["prompt_tokens"], msg["completion_tokens"] = len(prompt), len(ids)
        text = msg.get("content") or ""
        if after_tool and not msg.get("tool_calls") and text:
            # an answer written from tool results may open with the evidence line: it is taken off
            # the reply and checked against the tool result the model was shown last
            line, answer = convo.split_evidence(text)
            if line is not None:
                verdict = convo.check_evidence(line, format_v2.content_text(messages[-1].get("content")), answer)
                ok = verdict["ok"] if verdict["kind"] == "quote" else \
                    (looks_abstained(answer) if verdict["kind"] == "no_result" else True)
                msg["content"], msg["reasoning_content"], text = answer, line, answer
                msg["evidence"] = {"ok": bool(ok), "line": line}
                if not ok and self.evidence_check:
                    # the quote is not in the result it names (or the answer cites another result),
                    # or the model said that no result has the answer and answered anyway
                    msg["evidence"]["replaced"] = answer
                    msg["content"] = ABSTAIN_TEXT
                    return msg
        if tools and not msg.get("tool_calls") and CITES.search(text) and not _has_result_since_user(messages):
            # The model wrote citations without having searched (it happens on some questions, with a
            # right answer more often than not).  There is nothing behind those numbers.  If a search
            # tool is offered, this turn becomes the search the answer pretends to rest on, with the
            # user's words as the query; the model then answers from real results.
            spec = next((format_v2.tool_spec(t) for t in tools
                         if re.search(r"search|lookup|look_up|browse|wiki", format_v2.tool_spec(t).get("name", ""), re.I)), None)
            question = next((format_v2.content_text(m.get("content")) for m in reversed(messages) if m.get("role") == "user"), "")
            if spec is not None and question:
                args = json.dumps({_first_arg([spec]): question}, ensure_ascii=False)
                msg.update(content=None, finish_reason="tool_calls", forced_tool_call=text,
                           tool_calls=[{"id": "call_0", "type": "function", "function": {"name": spec["name"], "arguments": args}}])
                return msg
        if not msg.get("tool_calls") and CITES.search(text) and not _has_result_since_user(messages):
            # no search happened and none can be made (no search tool): the citation numbers and the
            # source list are made up, so they are taken off the reply
            text = strip_citations(text)
            msg["content"], msg["citations_removed"] = text, True
        if not msg.get("tool_calls") and text:
            fixed = plain_abstention(text, messages)
            if fixed != text:
                msg["content"], msg["abstention_reworded"], text = fixed, text, fixed
        if not self.self_check or msg.get("tool_calls") or not text or looks_abstained(text):
            return msg
        if after_tool:                                    # an answer written from tool results
            channel, threshold, fallback = 0, self.threshold, ABSTAIN_TEXT
        elif not tools and self.threshold_notools is not None:      # no tools in the request: from memory
            if ABOUT_SELF.search(text):                   # a reply about itself is never replaced
                return msg
            channel, threshold, fallback = 1, self.threshold_notools, DONT_KNOW_TEXT
        else:
            return msg
        # the head's output plus the model's own log-probabilities of what it wrote (the mean and the
        # lowest), with the weights that were fitted with the head and are stored next to the threshold
        a, b = self.lp_weight[channel]
        logit = self.engine.check(last, channel) + a * mean_lp + b * min_lp
        passed = logit >= threshold                       # the stored thresholds are logits
        # `margin` is the score minus the threshold: above 0 the reply is kept
        msg["self_check"] = {"margin": round(logit - threshold, 3), "passed": passed,
                             "kind": "after a search" if channel == 0 else "no tools"}
        if not passed:
            msg["self_check"]["replaced"] = text
            msg["content"] = fallback
        return msg

    # ---------------------------------------------------------------- the tool loop of the runners
    def ask(self, messages: List[Dict[str, Any]], search: Optional[Callable[[str], Sequence[Dict[str, Any]]]] = None,
            tools: Optional[List[Dict[str, Any]]] = None, max_tool_calls: int = 1,
            on_event: Optional[Callable[[str, Any], None]] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Answer the last user message, running `search(query)` for the model's search calls.

        `messages` is the conversation so far (it is not changed); returns (answer text, the new
        messages to append: tool calls, tool results and the final assistant message; the final
        message has `reasoning_content` when the answer opened with an evidence line).
        `on_event(kind, data)` kinds: tool_call, tool_result, forced_search, thinking (the evidence
        line), evidence (its verdict), self_check."""
        tools = tools if tools is not None else ([format_v2.WEB_SEARCH_TOOL] if search else None)
        new: List[Dict[str, Any]] = []
        calls = 0
        while True:
            send = list(messages) + new
            if self.history == "all":                          # "last" and "compact": complete() cuts it
                send = fit_history(send, tools, self.max_new_tokens, self.count)
            msg = self.complete(send, tools)
            if on_event and msg.get("reasoning_content"):
                on_event("thinking", msg["reasoning_content"])
            if on_event and msg.get("evidence"):
                on_event("evidence", msg["evidence"])
            if on_event and msg.get("self_check"):
                on_event("self_check", msg["self_check"])
            if on_event and msg.get("forced_tool_call"):
                on_event("forced_search", msg["forced_tool_call"])
            if not msg.get("tool_calls"):
                new.append({"role": "assistant", "content": msg.get("content") or ""})
                if msg.get("reasoning_content"):
                    new[-1]["reasoning_content"] = msg["reasoning_content"]
                return msg.get("content") or "", new
            new.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg["tool_calls"]})
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except ValueError:
                    args = {}
                if on_event:
                    on_event("tool_call", {"name": name, "arguments": args})
                if calls >= max_tool_calls:
                    content = "Error: no more tool calls are allowed for this question."
                elif search is None:
                    content = f"Error: tool '{name}' is not available."
                else:
                    query = next((v for v in args.values() if isinstance(v, str) and v.strip()), "")
                    try:
                        content = results_text(search(query)) if query else "No results."
                    except Exception as e:  # noqa: BLE001  a failed search is a tool result, not a crash
                        content = f"Error: search request failed ({type(e).__name__})."
                calls += 1
                if on_event:
                    on_event("tool_result", content)
                new.append({"role": "tool", "content": content})
            if calls > max_tool_calls + 1:                   # the model keeps calling: stop it
                new.append({"role": "assistant", "content": ABSTAIN_TEXT})
                return ABSTAIN_TEXT, new


import re  # noqa: E402

ABOUT_SELF = re.compile(r"kenchi|rex hall|rexmhall09|\u8b19\u77e5", re.I)
CITES = re.compile(r"\[\d+\]|^Sources:", re.M)


_ABOUT = re.compile(r"(?:anything|nothing|no source|no reliable source)\s+(?:about|for|on)\s+([^.:;,\n]{2,60})", re.I)


def plain_abstention(text: str, messages) -> str:
    """One of the trained ways of giving up names the thing it could not find ("I couldn't find
    anything about X.").  Sometimes the model uses that sentence with a name that is not in the
    conversation at all.  Such a reply is replaced by the plain sentence; everything else is kept."""
    m = _ABOUT.search(text or "")
    if not m or not looks_abstained(text):
        return text
    name = m.group(1).strip().lower()
    seen = " ".join(str(x.get("content") or "") for x in messages if x.get("role") in ("user", "tool", "tool_result")).lower()
    words = [w for w in re.findall(r"[^\W\d_]{3,}", name)]
    if words and not any(w in seen for w in words):
        return ABSTAIN_TEXT
    return text


def strip_citations(text: str) -> str:
    """The reply without `[n]` markers and without a `Sources:` block."""
    text = re.split(r"\n?^Sources:", text, maxsplit=1, flags=re.M)[0]
    text = re.sub(r"\s*\[\d+\]", "", text)
    return re.sub(r"\s+([.,;:])", r"\1", text).strip()


def _has_result_since_user(messages) -> bool:
    for m in reversed(messages):
        if m.get("role") in ("tool", "tool_result"):
            return True
        if m.get("role") == "user":
            return False
    return False


def _first_arg(tools) -> str:
    """The name of the first required argument of the first tool ("query" for the built-in search)."""
    try:
        params = format_v2.tool_spec(tools[0]).get("parameters") or {}
        return (params.get("required") or list((params.get("properties") or {"query": 0}).keys()))[0]
    except Exception:  # noqa: BLE001
        return "query"


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
