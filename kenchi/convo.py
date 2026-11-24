"""Conversation memory and the evidence line: pure text helpers shared by training data and runners.

Two conventions of the candidate after v2 (2026-11-21):

1. Compact history.  The model is shown the earlier turns of a conversation as plain exchanges:
   the user's text and the assistant's final answer, without the earlier tool calls, tool results,
   evidence lines, citation numbers and source lists.  Only the exchange being answered keeps its
   tool traffic.  `compact_history` builds that view; training data is built with the same function.

2. Evidence line ("thinking").  After a search the model first writes one line that says what it
   relies on, then the answer:

       [2] says: "written by English author Mary Shelley"
       The answer is Mary Shelley [2].
       Sources:
       [2] https://en.wikipedia.org/wiki/Frankenstein

   or, when no result has the answer:

       No result states this.
       I could not find a reliable source for that.

   The quote is a verbatim span (at most about 20 words) of result n.  `split_evidence` takes the
   line off the reply and `check_evidence` verifies it against the tool result.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

NO_EVIDENCE = "No result states this."
_EVID = re.compile(r'^\[(\d+)\] says: "(.*)"\s*$')
_CITE = re.compile(r"\s*\[\d+\]")


def evidence_line(n: int, quote: str) -> str:
    return f'[{int(n)}] says: "{" ".join(str(quote).split())}"'


def split_evidence(text: str) -> Tuple[Optional[str], str]:
    """-> (evidence line or None, the rest of the reply).  Only the first line can be one."""
    text = text or ""
    first, sep, rest = text.partition("\n")
    first = first.strip()
    if sep and (first == NO_EVIDENCE or _EVID.match(first)):
        return first, rest.lstrip("\n")
    return None, text


def _norm(s: str) -> str:
    return " ".join(str(s).split()).casefold()


def result_block(results_text: str, n: int) -> Optional[str]:
    """The text of result n in a tool result, for the layouts the runners and the training data use
    (`[n] Title - URL` + snippet line; `n. [Title](url)` + snippet; a JSON list).  None if not found."""
    text = results_text or ""
    s = text.lstrip()
    if s.startswith("[") and not re.match(r"\[\d+\]", s) or s.startswith("{"):
        try:
            import json
            data = json.loads(s)
            items = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
            if 1 <= n <= len(items):
                it = items[n - 1]
                return " ".join(str(v) for v in it.values()) if isinstance(it, dict) else str(it)
        except ValueError:
            pass
    starts = [(int(m.group(1) or m.group(2)), m.start()) for m in re.finditer(r"(?m)^(?:\[(\d+)\] |(\d+)[.)] )", text)]
    for i, (num, pos) in enumerate(starts):
        if num == n:
            end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
            return text[pos:end]
    return None


def check_evidence(line: Optional[str], results_text: str, answer: str = "") -> Dict[str, Any]:
    """Is the evidence line true to the tool result?

    -> {"kind": "none" | "quote" | "no_result", "ok": bool, "n": cited result or None}
    "quote": ok when the quoted words are in result n (or, when the result layout is not one we can
    split, anywhere in the tool result) and the answer cites n when it cites anything.
    "no_result": always ok (the runner decides what to do when the answer is not a give-up)."""
    if not line:
        return {"kind": "none", "ok": True, "n": None}
    if line.strip() == NO_EVIDENCE:
        return {"kind": "no_result", "ok": True, "n": None}
    m = _EVID.match(line.strip())
    if not m:
        return {"kind": "none", "ok": True, "n": None}
    n, quote = int(m.group(1)), _norm(m.group(2)).strip(" .…")
    block = result_block(results_text, n)
    hay = _norm(block if block is not None else results_text)
    ok = len(quote) >= 3 and quote in hay
    cited = [int(x) for x in re.findall(r"\[(\d+)\]", (answer or "").split("\nSources")[0])]
    if ok and cited and n not in cited:
        ok = False
    return {"kind": "quote", "ok": ok, "n": n}


def clean_answer(text: str) -> str:
    """An earlier answer as it is shown in the history: no evidence line, no `[n]`, no source list."""
    _line, rest = split_evidence(text or "")
    rest = re.split(r"\n?^Sources:", rest, maxsplit=1, flags=re.M)[0]
    rest = _CITE.sub("", rest)
    return re.sub(r"\s+([.,;:])", r"\1", " ".join(rest.split())).strip()


def compact_history(messages: List[Dict[str, Any]], count: Optional[Callable[[List[Dict[str, Any]]], int]] = None,
                    budget: int = 400) -> List[Dict[str, Any]]:
    """The conversation as the model is shown it.

    Leading system messages stay.  Every exchange before the last user message becomes
    `user text` + `assistant final answer` (clean_answer); exchanges without a final answer are
    dropped.  The last exchange (the last user message and everything after it) is kept as it is.
    With `count` (messages -> tokens), the oldest earlier exchanges are dropped until the earlier
    part is at most `budget` tokens."""
    msgs = list(messages)
    last = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=None)
    if last is None:
        return msgs
    n = 0
    while n < last and msgs[n].get("role") in ("system", "developer"):
        n += 1
    head, earlier, tail = msgs[:n], msgs[n:last], msgs[last:]
    pairs: List[List[Dict[str, Any]]] = []
    user = None
    answer = None
    for m in earlier + [{"role": "user", "content": None}]:           # the sentinel closes the last exchange
        role = m.get("role")
        if role == "user":
            if user is not None and answer:
                pairs.append([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
            user, answer = _text(m.get("content")), None
        elif role == "assistant" and not m.get("tool_calls") and m.get("content"):
            answer = clean_answer(_text(m.get("content")))
    if count is not None:
        while pairs and count([x for p in pairs for x in p]) > budget:
            pairs.pop(0)
    return head + [x for p in pairs for x in p] + tail


def _text(content) -> str:
    if isinstance(content, list):                                      # OpenAI content parts
        return " ".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return "" if content is None else str(content)
