"""Kenchi chat format, version 2: the one place that turns messages into tokens and back.

Harnesses speak the OpenAI chat format: a list of messages (roles `system`,
`user`, `assistant`, `tool`), an optional `tools` list of JSON schemas, and
assistant messages that may carry `tool_calls`.  The model sees a token stream:

    <|endoftext|><|system|>SYSTEM<|end|>
    <|user|>text<|end|>
    <|assistant|>text<|end|>
    <|assistant|><|tool_call|>{"name": "web_search", "arguments": {"query": "..."}}<|end|>
    <|tool_result|>whatever the tool returned, as text<|end|>
    ...
    <|assistant|>                      <- the model continues from here

SYSTEM is the harness's system message (or DEFAULT_SYSTEM when there is none),
followed, when tools are given, by a blank line, the line `Tools:` and one line
of JSON per tool: `{"name": ..., "description": ..., "parameters": {...}}`.

Several tool calls in one assistant message are written as one JSON list in one
`<|tool_call|>` turn.  JSON is always `json.dumps(x, ensure_ascii=False)` with
Python's default separators, keys in the order given.

Training data, the runners, the Hugging Face chat template and the server all
go through `render` / `render_text`, so what the model is trained on and what
it sees in use are the same bytes (tests/test_format_v2.py).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

ENDOFTEXT, SYSTEM, USER, ASSISTANT, TOOL_CALL, TOOL_RESULT, END = range(50256, 50263)
SPECIAL = {ENDOFTEXT: "<|endoftext|>", SYSTEM: "<|system|>", USER: "<|user|>", ASSISTANT: "<|assistant|>",
           TOOL_CALL: "<|tool_call|>", TOOL_RESULT: "<|tool_result|>", END: "<|end|>"}

DEFAULT_SYSTEM = "You are Kenchi, a small language model made by Rex Hall (rexmhall09)."

#: The search tool of the built-in runners (run.py, run_mlx.py).
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web and return the top results.",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "description": "The search query."}},
                   "required": ["query"]},
}


def dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def tool_spec(tool: Dict[str, Any]) -> Dict[str, Any]:
    """An OpenAI tool entry (`{"type": "function", "function": {...}}`) or a bare
    function object -> `{"name", "description", "parameters"}` (absent keys left out)."""
    f = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    out = {"name": f.get("name", "")}
    if f.get("description"):
        out["description"] = f["description"]
    if f.get("parameters") is not None:
        out["parameters"] = f["parameters"]
    return out


def system_text(system: Optional[str], tools: Optional[Sequence[Dict[str, Any]]]) -> str:
    text = DEFAULT_SYSTEM if system is None else str(system)
    if tools:
        block = "Tools:\n" + "\n".join(dumps(tool_spec(t)) for t in tools)
        text = (text + "\n\n" + block) if text else block
    return text


def content_text(content: Any) -> str:
    """Message content as text: a string, or OpenAI content parts (their text joined)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(str(p.get("text") or p.get("content") or ""))
            else:
                parts.append(str(p))
        return "".join(parts)
    return dumps(content) if isinstance(content, (dict, int, float, bool)) else str(content)


def call_text(tool_calls: Sequence[Dict[str, Any]]) -> str:
    """The body of a `<|tool_call|>` turn for OpenAI `tool_calls` (or bare `{"name","arguments"}` calls)."""
    calls = []
    for c in tool_calls:
        f = c.get("function") if isinstance(c.get("function"), dict) else c
        args = f.get("arguments", {})
        if isinstance(args, str):                       # OpenAI sends arguments as a JSON string
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {"_raw": args}
        calls.append({"name": f.get("name", ""), "arguments": args})
    return dumps(calls[0] if len(calls) == 1 else calls)


Segment = Tuple[Optional[int], str]      # (special token id, "") or (None, text)


def segments(messages: Sequence[Dict[str, Any]], tools: Optional[Sequence[Dict[str, Any]]] = None,
             add_generation_prompt: bool = True) -> List[Segment]:
    """The conversation as special tokens and text pieces, in order."""
    msgs = list(messages)
    system = None
    if msgs and msgs[0].get("role") in ("system", "developer"):
        system = content_text(msgs[0].get("content"))
        msgs = msgs[1:]
    seg: List[Segment] = [(ENDOFTEXT, ""), (SYSTEM, ""), (None, system_text(system, tools)), (END, "")]
    for m in msgs:
        role = m.get("role")
        if role in ("system", "developer"):             # a later system message: shown as a user turn
            seg += [(USER, ""), (None, content_text(m.get("content"))), (END, "")]
        elif role == "user":
            seg += [(USER, ""), (None, content_text(m.get("content"))), (END, "")]
        elif role == "assistant":
            text = content_text(m.get("content"))
            calls = m.get("tool_calls") or []
            if text or not calls:
                seg += [(ASSISTANT, ""), (None, text), (END, "")]
            if calls:
                seg += [(ASSISTANT, ""), (TOOL_CALL, ""), (None, call_text(calls)), (END, "")]
        elif role in ("tool", "function", "tool_result"):
            seg += [(TOOL_RESULT, ""), (None, content_text(m.get("content"))), (END, "")]
        elif role == "tool_call":                       # Kenchi's own older role name
            seg += [(ASSISTANT, ""), (TOOL_CALL, ""), (None, content_text(m.get("content"))), (END, "")]
        else:
            raise ValueError(f"unknown role {role!r}")
    if add_generation_prompt:
        seg.append((ASSISTANT, ""))
    return seg


def render_text(messages, tools=None, add_generation_prompt=True) -> str:
    """The prompt as one string with the special tokens spelled out (what the HF chat template gives)."""
    return "".join(SPECIAL[t] if t is not None else s for t, s in segments(messages, tools, add_generation_prompt))


def render(messages, tools=None, add_generation_prompt=True, encode=None) -> List[int]:
    """The prompt as token ids. `encode(text) -> ids` is the plain-text tokenizer
    (default: kenchi.tokenizer.encode_text)."""
    if encode is None:
        from kenchi import tokenizer as _t
        encode = _t.encode_text
    ids: List[int] = []
    for t, s in segments(messages, tools, add_generation_prompt):
        if t is not None:
            ids.append(t)
        elif s:
            ids.extend(int(x) for x in encode(s))
    return ids


def parse_output(ids: Sequence[int], decode=None) -> Dict[str, Any]:
    """Generated ids (after the `<|assistant|>` of the prompt) -> an OpenAI assistant message.

    Returns `{"role": "assistant", "content": str or None, "tool_calls": [...]}` plus
    `"finish_reason"` (`"tool_calls"`, `"stop"` or `"length"`) and, for a tool-call turn
    whose JSON cannot be read, `"invalid_tool_call": raw_text` with the text as content.
    """
    if decode is None:
        from kenchi import tokenizer as _t
        decode = _t.decode_text
    ids = [int(t) for t in ids]
    ended = END in ids
    if ended:
        ids = ids[: ids.index(END)]
    is_call = bool(ids) and ids[0] == TOOL_CALL
    body = ids[1:] if is_call else ids
    for n, t in enumerate(body):                        # nothing after another turn marker counts
        if t >= ENDOFTEXT:
            body = body[:n]
            ended = True
            break
    text = decode(body)
    msg: Dict[str, Any] = {"role": "assistant", "content": text, "finish_reason": "stop" if ended else "length"}
    if not is_call:
        return msg
    try:
        obj = json.loads(text)
        calls = obj if isinstance(obj, list) else [obj]
        out = []
        for i, c in enumerate(calls):
            name, args = c["name"], c.get("arguments", {})
            if not isinstance(name, str) or not name or not isinstance(args, dict):
                raise ValueError("bad call")
            out.append({"id": f"call_{i}", "type": "function",
                        "function": {"name": name, "arguments": dumps(args)}})
        if not out:
            raise ValueError("no calls")
        msg.update(content=None, tool_calls=out, finish_reason="tool_calls")
    except (ValueError, KeyError, TypeError):
        msg["invalid_tool_call"] = text
    return msg
