"""Structured turns + rendering per docs/CHAT_FORMAT.md (binding).

Structured turns are dicts:

- system / user / assistant: ``{"role": ..., "text": str}``
- tool call:                 ``{"role": "tool_call", "name": str, "arguments": dict}``
- tool result:               ``{"role": "tool_result", "text": str}``

`render_text(turns, system)` returns the document string and
`render_tokens(turns, system)` the token stream, per the doc:

``<|system|>{system}<|end|><|user|>{q}<|end|><|assistant|><|tool_call|>{json}<|end|>``
... — no newlines between turns, and a tool call is an assistant turn
containing only one JSON object ``{"name", "arguments"}``.

The token stream is produced with `kenchi.chat_format.encode` (the
repo's tokeniser helper) and the special ids of the doc.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kenchi.chat_format import (  # noqa: E402
    DEFAULT_SYSTEM_TEXT,
    TOKEN_ASSISTANT,
    TOKEN_END,
    TOKEN_SYSTEM,
    TOKEN_TOOL_CALL,
    TOKEN_TOOL_RESULT,
    TOKEN_USER,
)

try:  # W10-to-W11 tokenizer switch: the new multilingual tokenizer when available.
    from kenchi.tokenizer_core import load_tokenizer  # noqa: E402

    _KENCHI_TOKENIZER = load_tokenizer()

    def encode(text):  # noqa: E402
        """Token IDs via the new Kenchi tokenizer (W11)."""
        return _KENCHI_TOKENIZER.encode(text)
except Exception:  # noqa: BLE001 - falls back to the GPT-2 helper until the
    # kenchi_sp.model file lands.
    from kenchi.chat_format import encode  # noqa: E402,F401

#: The FineWeb-Edu document separator (docs/CHAT_FORMAT.md: id 50256 =
#: ``<|endoftext|>``, "start of every document / conversation").
TOKEN_ENDOFTEXT = 50256

__all__ = [
    "DEFAULT_SYSTEM_TEXT",
    "NO_TOOL_SYSTEM_TEXT",
    "TOKEN_ENDOFTEXT",
    "TOKEN_ASSISTANT",
    "TOKEN_END",
    "TOKEN_SYSTEM",
    "TOKEN_TOOL_CALL",
    "TOKEN_TOOL_RESULT",
    "TOKEN_USER",
    "TOOL_NAMES",
    "assistant_turn",
    "cites_of",
    "format_open_page_result",
    "format_results",
    "parse_citations",
    "render_text",
    "render_tokens",
    "result_turn",
    "system_turn",
    "tool_call_turn",
    "tool_result_turn",
    "user_turn",
]

#: Tools declared by the Kenchi chat format (see docs/CHAT_FORMAT.md).
TOOL_NAMES = ("web_search", "open_page")

#: The default system text without the tool sentences ("the same text without
#: the tool sentences" per docs/CHAT_FORMAT.md).
NO_TOOL_SYSTEM_TEXT = DEFAULT_SYSTEM_TEXT.replace(
    " You can use tools: web_search(query) and open_page(url).", ""
)


def system_turn(text: str = DEFAULT_SYSTEM_TEXT) -> Dict[str, Any]:
    return {"role": "system", "text": text}


def user_turn(text: str) -> Dict[str, Any]:
    return {"role": "user", "text": text}


def assistant_turn(text: str) -> Dict[str, Any]:
    return {"role": "assistant", "text": text}


def tool_call_turn(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {"role": "tool_call", "name": name, "arguments": arguments}


def result_turn(text: str) -> Dict[str, Any]:
    return {"role": "tool_result", "text": text}


def tool_result_turn(results, golds=(), offset: int = 0) -> Dict[str, Any]:
    """The tool-result turn for a `web_search` result list (W15b: numbers
    start at offset+1 — cumulative numbering)."""
    return result_turn(format_results(results, golds=golds, offset=offset))


def _call_json(name: str, arguments: Dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments})


def _truncate_words(text: str, n_words: int) -> str:
    words = (text or "").split()
    return " ".join(words[:n_words])


def format_results(results, snippet_words: int = 60, golds=(), offset: int = 0) -> str:
    """Render search results as the runner does: ``[n] Title - URL`` lines
    followed by the snippet, for each result; ``No results.`` when empty.

    Per docs/CHAT_FORMAT.md the snippet is "about 60 words"; snippets are
    truncated to `snippet_words` words unless they contain a gold answer
    (the citation gate needs the gold present in the cited result text).

    Result numbers run CUMULATIVELY through a conversation (W15b): the
    first search returns [offset+1]..[offset+len], so pass `offset` = the
    number of results already numbered before this block.
    """
    from .normalise import norm_contains

    blocks = []
    for i, r in enumerate(results or [], offset + 1):
        snippet = r.get("snippet") or ""
        if not any(
            norm_contains(f"{r.get('title', '')} {snippet}", g) for g in (golds or ())
        ):
            snippet = _truncate_words(snippet, snippet_words)
        blocks.append(f"[{i}] {r['title']} - {r['url']}\n{snippet}")
    return "\n".join(blocks) if blocks else "No results."


def format_open_page_result(title: str, url: str, text: str, max_words: int = 350) -> str:
    """Render an `open_page` result: ``[page] Title - URL`` + the page text,
    truncated to about 350 words per docs/CHAT_FORMAT.md."""
    return f"[page] {title} - {url}\n{_truncate_words(text, max_words)}"


def parse_citations(text: str) -> List[int]:
    """Result numbers cited in `text`, in order of appearance."""
    return [int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", text or "")]


def result_urls_from_block(block_text: str) -> Dict[int, str]:
    """Map result number -> URL from a `web_search` result block
    (``[n] Title - URL`` lines); the URL is copied verbatim from the block."""
    out: Dict[int, str] = {}
    for line in (block_text or "").split("\n"):
        m = re.match(r"^\[(\d+)\]\s?.*\s(https?://\S+)$", line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def sources_block(nums, urls: Dict[int, str]) -> str:
    """Sources block lines ``[n] URL`` for `nums` (URLs from
    `result_urls_from_block` or `page["url"]`)."""
    lines = [f"[{n}] {urls[n]}" for n in nums if n in urls]
    return "\nSources:\n" + "\n".join(lines) if lines else ""


def cites_of(numbers: Sequence[int]) -> str:
    """Render result numbers as citations, e.g. ``[1] [3]``."""
    return " ".join(f"[{n}]" for n in numbers)


def render_text(turns: Sequence[Dict[str, Any]], system: str = DEFAULT_SYSTEM_TEXT) -> str:
    """Render structured turns to the document string (no newlines). A leading
    system turn in `turns` supplies the system text; `system` is the fallback."""
    parts: List[str] = ["<|endoftext|>"]
    i = 0
    if turns and turns[0].get("role") == "system":
        parts.append(f"<|system|>{turns[0]['text']}<|end|>")
        i = 1
    else:
        parts.append(f"<|system|>{system}<|end|>")
    for t in turns[i:]:
        r = t["role"]
        if r == "user":
            parts.append(f"<|user|>{t['text']}<|end|>")
        elif r == "assistant":
            parts.append(f"<|assistant|>{t['text']}<|end|>")
        elif r == "tool_call":
            parts.append(f"<|assistant|><|tool_call|>{_call_json(t['name'], t['arguments'])}<|end|>")
        elif r == "tool_result":
            parts.append(f"<|tool_result|>{t['text']}<|end|>")
        else:
            raise ValueError(f"unknown role {r!r}")
    return "".join(parts)


def render_tokens(turns: Sequence[Dict[str, Any]], system: str = DEFAULT_SYSTEM_TEXT):
    """Token stream per docs/CHAT_FORMAT.md (numpy uint16 array). The
    conversation document starts with token 50256 (`<|endoftext|>`)."""
    import numpy as np

    toks: List[int] = [TOKEN_ENDOFTEXT]
    i = 0
    if turns and turns[0].get("role") == "system":
        toks += [TOKEN_SYSTEM] + encode(turns[0]["text"]) + [TOKEN_END]
        i = 1
    else:
        toks += [TOKEN_SYSTEM] + encode(system) + [TOKEN_END]
    for t in turns[i:]:
        r = t["role"]
        if r == "user":
            toks += [TOKEN_USER] + encode(t["text"]) + [TOKEN_END]
        elif r == "assistant":
            toks += [TOKEN_ASSISTANT] + encode(t["text"]) + [TOKEN_END]
        elif r == "tool_call":
            toks += [TOKEN_ASSISTANT, TOKEN_TOOL_CALL] + encode(_call_json(t["name"], t["arguments"])) + [TOKEN_END]
        elif r == "tool_result":
            toks += [TOKEN_TOOL_RESULT] + encode(t["text"]) + [TOKEN_END]
        else:
            raise ValueError(f"unknown role {r!r}")
    return np.asarray(toks, dtype=np.uint16)
