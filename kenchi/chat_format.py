"""Chat rendering per docs/CHAT_FORMAT.md (the Kenchi tokenizer + special ids 50256-50262).

Rendering for an instruction conversation (system turn + alternating
user/assistant turns):

    <|endoftext|><|system|>{system text}<|end|>
    <|user|>{user text}<|end|>
    <|assistant|>{assistant text}<|end|>

There are no newlines between turns; every conversation document starts with
token 50256 (the FineWeb-Edu document separator), and ids 50257-50262 are
emitted as special ids while the text segments are encoded with tiktoken's
GPT-2 BPE.
"""

from __future__ import annotations

import numpy as np

DOC_SEP = 50256

TOKEN_ENDOFTEXT = 50256
TOKEN_SYSTEM = 50257
TOKEN_USER = 50258
TOKEN_ASSISTANT = 50259
TOKEN_TOOL_CALL = 50260
TOKEN_TOOL_RESULT = 50261
TOKEN_END = 50262

DEFAULT_SYSTEM_TEXT = (
    "You are Kenchi, a small language model made by Rex Hall (rexmhall09). You can use tools: "
    "web_search(query) and open_page(url). Search when you are not sure. Answer briefly and cite "
    "sources like [1]. If you do not know or cannot find a reliable source, say so."
)


def encode(text):
    # disallowed_special=() so literal "<|endoftext|>" in corpus text is
    # encoded as normal text instead of raising.
    import tiktoken  # only the Phase 1 (GPT-2 BPE) path needs it

    return tiktoken.get_encoding("gpt2").encode(text, disallowed_special=())


def render_conversation(turns, system=DEFAULT_SYSTEM_TEXT, include_system=True):
    """Render a conversation per docs/CHAT_FORMAT.md.

    `turns`: list of (role, text) tuples in order; role in
    {"user", "assistant", "system", "tool_call", "tool_result"}.

    Returns a uint16 token array starting with the system turn and ending
    with `<|end|>` (the FineWeb-Edu document separator 50256 is written by the
    shard writer, so the conversation itself starts right after it).
    """
    toks = []
    if include_system:
        toks += [TOKEN_SYSTEM] + encode(system) + [TOKEN_END]
    for role, text in turns:
        if role == "user":
            toks += [TOKEN_USER]
        elif role == "assistant":
            toks += [TOKEN_ASSISTANT]
        elif role == "system":
            toks += [TOKEN_SYSTEM]
        elif role == "tool_call":
            toks += [TOKEN_TOOL_CALL]
        elif role == "tool_result":
            toks += [TOKEN_TOOL_RESULT]
        else:
            raise ValueError(f"unknown role {role!r}")
        toks += encode(text) + [TOKEN_END]
    return np.asarray(toks, dtype=np.uint16)
