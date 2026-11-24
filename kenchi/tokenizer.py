"""Chat-format tokenizer on the Kenchi core (src/kenchi/tokenizer.py).

The canonical core is ``kenchi.tokenizer_core``: ``load_tokenizer()`` returns
the trained Kenchi tokenizer (SentencePiece unigram, byte fallback, pieces at
ids 0..50255; special ids of docs/CHAT_FORMAT.md at 50256..50262, model
vocab_size 50304). Until the model file lands, a thin tiktoken-gpt2 fallback
with the same interface is used (``tokenizer_kind() == "tiktoken-fallback"``);
once it lands, nothing here changes (``tokenizer_kind() == "sentencepiece"``).

On top of the core, this module provides the conversation-level API the
runner, SFT and calibration share:

- encode_text / decode_text: plain-text encode/decode via the core.
- encode_conversation(turns, system=None): the token-id stream of a structured
  conversation per docs/CHAT_FORMAT.md (the doc separator + system turn +
  turns, built with the core's encode, identical in shape to
  kenchi_sft.chat.render_tokens).
- decode_conversation(ids): the inverse -- scans the stream into structured
  turns (system / user / assistant / tool_call / tool_result).
- assistant_mask(ids): for a rendered conversation, the boolean mask of
  ASSISTANT-turn tokens: everything after `<|assistant|>` up to and
  including its `<|end|>`.

Round trip: encode_conversation(decode_conversation(stream)) == stream for
the conversation streams in the shipped SFT shards (checked on the first 200
stored examples in tests/test_chat.py while the GPT-2 fallback is active;
with the trained tokenizer the round trip is checked on synthetic
conversations, since the shards predate the switch).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from kenchi import tokenizer_core as _core_mod
from kenchi import render as kchat
from kenchi.chat_format import (
    TOKEN_ASSISTANT,
    TOKEN_END,
    TOKEN_ENDOFTEXT,
    TOKEN_SYSTEM,
    TOKEN_TOOL_CALL,
    TOKEN_TOOL_RESULT,
    TOKEN_USER,
)

__all__ = ["assistant_mask", "decode_conversation", "decode_text",
           "encode_conversation", "encode_text", "tokenizer_kind"]

class _TiktokenFallback:
    """Thin fallback used until ``~/kenchi_data/tokenizer/kenchi_sp.model``
    lands: tiktoken ``gpt2`` with the same interface as the core's
    ``KenchiTokenizer`` (the interface tokenizer.py programs against)."""

    kind = "tiktoken-fallback"
    is_sentencepiece = False

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")
        self.TOKEN_ENDOFTEXT = TOKEN_ENDOFTEXT
        self.TOKEN_SYSTEM = TOKEN_SYSTEM
        self.TOKEN_USER = TOKEN_USER
        self.TOKEN_ASSISTANT = TOKEN_ASSISTANT
        self.TOKEN_TOOL_CALL = TOKEN_TOOL_CALL
        self.TOKEN_TOOL_RESULT = TOKEN_TOOL_RESULT
        self.TOKEN_END = TOKEN_END

    def encode(self, text: str) -> List[int]:
        return [int(t) for t in self._enc.encode(text, disallowed_special=())]

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]

    def decode(self, ids) -> str:
        # The special range (>= 50256) has no text encoding; drop it (the
        # caller wants the text of the stream body).
        return self._enc.decode([int(t) for t in ids if 0 <= int(t) < TOKEN_ENDOFTEXT])


_CORE = None


def _core():
    global _CORE
    if _CORE is None:
        try:
            _CORE = _core_mod.load_tokenizer()
        except FileNotFoundError:
            # The trained model has not landed; the tests and the runner run
            # on the tiktoken fallback (same interface).
            _CORE = _TiktokenFallback()
    return _CORE


def tokenizer_kind() -> str:
    """``"sentencepiece"`` (trained tokenizer) or ``"tiktoken-fallback"``."""
    return "sentencepiece" if getattr(_core(), "is_sentencepiece", False) \
        else "tiktoken-fallback"


# ------------------------------------------------------------------ text

def encode_text(text: str) -> List[int]:
    """Plain-text ids via the core."""
    return _core().encode(text)


def decode_text(ids) -> str:
    """Decode ids to text; the special range (>= 50256) has no text encoding
    and is dropped (the runner needs the text of arbitrary model outputs)."""
    toks = [int(t) for t in ids if 0 <= int(t) < TOKEN_ENDOFTEXT]
    return _core().decode(toks) if toks else ""


# ------------------------------------------------------------------ conversation

def _call_json(name: str, arguments: Dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments})


def _render_tokens_with_core(turns, system) -> List[int]:
    """The stream kenchi_sft.chat.render_tokens emits, with the core's encode:
    ``[50256, 50257, encode(system), 50262, turns...]`` (no newlines between
    turns; a tool call is an assistant turn with only the call JSON)."""
    enc = _core().encode
    toks: List[int] = [TOKEN_ENDOFTEXT]
    i = 0
    if turns and turns[0].get("role") == "system":
        toks += [TOKEN_SYSTEM] + enc(turns[0]["text"]) + [TOKEN_END]
        i = 1
    else:
        toks += [TOKEN_SYSTEM] + enc(system) + [TOKEN_END]
    for t in turns[i:]:
        r = t["role"]
        if r == "user":
            toks += [TOKEN_USER] + enc(t["text"]) + [TOKEN_END]
        elif r == "assistant":
            toks += [TOKEN_ASSISTANT] + enc(t["text"]) + [TOKEN_END]
        elif r == "tool_call":
            toks += [TOKEN_ASSISTANT, TOKEN_TOOL_CALL] + enc(_call_json(t["name"], t["arguments"])) + [TOKEN_END]
        elif r == "tool_result":
            toks += [TOKEN_TOOL_RESULT] + enc(t["text"]) + [TOKEN_END]
        else:
            raise ValueError(f"unknown role {r!r}")
    return toks


def encode_conversation(turns: Sequence[Dict[str, Any]],
                        system: Optional[str] = None) -> List[int]:
    """Token stream of `turns` per docs/CHAT_FORMAT.md."""
    turns = list(turns)
    sys_text = kchat.DEFAULT_SYSTEM_TEXT if system is None else system
    if tokenizer_kind() == "sentencepiece":
        return [int(t) for t in _render_tokens_with_core(turns, sys_text)]
    rt = kchat.render_tokens(turns, sys_text)
    return [TOKEN_ENDOFTEXT] + [int(t) for t in rt.tolist()]


def decode_conversation(ids) -> List[Dict[str, Any]]:
    """Inverse of encode_conversation: scan the stream into structured turns."""
    toks = [int(t) for t in ids]
    while toks and toks[0] == TOKEN_ENDOFTEXT:
        toks.pop(0)
    turns: List[Dict[str, Any]] = []
    n = len(toks)
    i = 0
    if i < n and toks[i] == TOKEN_SYSTEM:
        text, i = _read_until_end(toks, i + 1)
        turns.append({"role": "system", "text": text})
    while i < n:
        t = toks[i]
        if t == TOKEN_USER:
            text, i = _read_until_end(toks, i + 1)
            turns.append({"role": "user", "text": text})
        elif t == TOKEN_ASSISTANT:
            i += 1
            if i < n and toks[i] == TOKEN_TOOL_CALL:
                raw, i = _read_until_end(toks, i + 1)
                try:
                    obj = json.loads(raw) if raw.strip() else {}
                except ValueError:
                    obj = {}
                turns.append({"role": "tool_call", "name": obj.get("name"),
                              "arguments": obj.get("arguments") or {}})
            else:
                text, i = _read_until_end(toks, i)
                turns.append({"role": "assistant", "text": text})
        elif t == TOKEN_TOOL_RESULT:
            text, i = _read_until_end(toks, i + 1)
            turns.append({"role": "tool_result", "text": text})
        else:
            break
    return turns


def _read_until_end(toks, j):
    """Decode text from `j` until the next special id; returns (text, index
    past the `<|end|>` that terminated the read, or `j` if none)."""
    out = []
    n = len(toks)
    while j < n and toks[j] < TOKEN_ENDOFTEXT:
        out.append(toks[j])
        j += 1
    text = decode_text(out)
    if j < n and toks[j] == TOKEN_END:
        j += 1
    return text, j


def assistant_mask(ids) -> List[bool]:
    """Boolean mask of ASSISTANT-turn tokens of a rendered conversation.

    True for everything after a `<|assistant|>` up to and including the
    `<|end|>` of that turn (so tool-call turns are covered: the
    `<|tool_call|>` token follows `<|assistant|>`).
    """
    toks = [int(t) for t in ids]
    mask = [False] * len(toks)
    i = 0
    n = len(toks)
    while i < n:
        if toks[i] == TOKEN_ASSISTANT:
            j = i + 1
            while j < n:
                mask[j] = True
                if toks[j] == TOKEN_END:
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return mask
