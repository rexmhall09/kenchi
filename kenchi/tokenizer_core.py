"""Kenchi multilingual tokenizer core (src/kenchi/tokenizer_core.py).

SentencePiece-backed implementation of the interface of docs/CHAT_FORMAT.md
and of WORK PACKAGE W14:

* ``KenchiTokenizer`` with ``encode(text) -> list[int]`` (ordinary text only;
  never emits ids >= 50256), ``decode(ids, skip_special=False) -> str``
  (chat-format specials render as their literal strings by default, or are
  skipped with ``skip_special=True``), and
  ``encode_batch(list[str]) -> list[list[int]]``.
* Special-id attributes: ``TOKEN_ENDOFTEXT_ID`` (50256) ... ``TOKEN_END_ID``
  (50262), ``SPECIAL_IDS``, ``SPECIAL_STRINGS``; vocabulary attributes
  ``vocab_size = 50304`` and ``n_base = 50256``.
* ``load_tokenizer(path=None)``: ``$KENCHI_TOKENIZER``, then
  ``~/kenchi_data/tokenizer/kenchi_sp.model``, then the packaged copy
  ``src/kenchi/assets/kenchi_sp.model``; raises ``FileNotFoundError`` when
  none exists.

Newlines and tabs are pre-mapped to private-use characters (U+E000, U+E001)
before encoding and mapped back after decoding, so text round-trips exactly
(the trainer learned pieces for them; see scripts/train_tokenizer.py and
notes/tokenizer.md). The wrapper is thread-safe and picklable (the model is
carried in the pickle payload).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence

import sentencepiece as spm

#: Ids: the model has 50,256 base pieces (0..50255); specials keep their
#: docs/CHAT_FORMAT.md ids; ids 50263..50303 stay reserved.
N_BASE = 50256
VOCAB_SIZE = 50304

TOKEN_ENDOFTEXT_ID = 50256   # <|endoftext|>
TOKEN_SYSTEM_ID = 50257      # <|system|>
TOKEN_USER_ID = 50258        # <|user|>
TOKEN_ASSISTANT_ID = 50259   # <|assistant|>
TOKEN_TOOL_CALL_ID = 50260   # <|tool_call|>
TOKEN_TOOL_RESULT_ID = 50261  # <|tool_result|>
TOKEN_END_ID = 50262         # <|end|>

SPECIAL_STRINGS = {
    TOKEN_ENDOFTEXT_ID: "<|endoftext|>",
    TOKEN_SYSTEM_ID: "<|system|>",
    TOKEN_USER_ID: "<|user|>",
    TOKEN_ASSISTANT_ID: "<|assistant|>",
    TOKEN_TOOL_CALL_ID: "<|tool_call|>",
    TOKEN_TOOL_RESULT_ID: "<|tool_result|>",
    TOKEN_END_ID: "<|end|>",
}
SPECIAL_IDS = tuple(SPECIAL_STRINGS)
PIECES_MAX = N_BASE      # stub name kept for compat: ids 0..50255 are pieces

_NL = "\ue000"   # private-use newline inside the base vocabulary
_TAB = "\ue001"  # private-use tab inside the base vocabulary

TOKDIR = Path.home() / "kenchi_data" / "tokenizer"
ASSET_PATH = Path(__file__).resolve().parent / "assets" / "kenchi_sp.model"


def _map_in(text: str) -> str:
    # Literal U+E000/U+E001 in user text cannot round-trip as themselves
    # (they are the newline/tab pieces): encode maps them to "\n"/"\t"
    # deterministically, so decode(encode(t)) yields the newline/tab.
    # Documented in notes/tokenizer.md ("Known limits").
    return (text.replace("\ue000", "\n").replace("\ue001", "\t")
            .replace("\n", _NL).replace("\t", _TAB))


def _map_out(text: str) -> str:
    return text.replace(_NL, "\n").replace(_TAB, "\t")


class _Gpt2Fallback:
    """Optional GPT-2 BPE fallback (stub name kept for compat).

    Used only when ``KENCHI_TOKENIZER=gpt2`` is set explicitly; the real
    SentencePiece tokenizer is the default once the W14 model file exists.
    """

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        return [min(int(i), PIECES_MAX - 1) for i in self._enc.encode(text)]

    def decode(self, ids) -> str:
        return self._enc.decode([int(i) for i in ids if 0 <= int(i) < PIECES_MAX])

    def encode_batch(self, texts) -> List[List[int]]:
        return [self.encode(t) for t in texts]


class KenchiTokenizer:
    """SentencePiece tokenizer over the Kenchi base vocabulary (see module doc)."""

    N_BASE = N_BASE
    VOCAB_SIZE = VOCAB_SIZE
    EOS_ID = TOKEN_ENDOFTEXT_ID
    SYSTEM_ID = TOKEN_SYSTEM_ID
    USER_ID = TOKEN_USER_ID
    ASSISTANT_ID = TOKEN_ASSISTANT_ID
    TOOL_CALL_ID = TOKEN_TOOL_CALL_ID
    TOOL_RESULT_ID = TOKEN_TOOL_RESULT_ID
    END_ID = TOKEN_END_ID

    def __init__(self, sp_model: Any = None):
        if sp_model is None:
            if os.environ.get("KENCHI_TOKENIZER") == "gpt2":
                sp_model = _Gpt2Fallback()
            else:
                sp_model = _find_model_path()
        if isinstance(sp_model, (str, Path)):
            self.model_file = Path(sp_model)
            self._sp = _load_sp(self.model_file)
        else:
            self.model_file = None
            self._sp = sp_model
        if self._sp is None:
            raise RuntimeError("KenchiTokenizer: no SentencePiece model loaded")
        #: Special-id attributes of docs/CHAT_FORMAT.md.
        self.TOKEN_ENDOFTEXT = self.EOS_ID
        self.TOKEN_SYSTEM = self.SYSTEM_ID
        self.TOKEN_USER = self.USER_ID
        self.TOKEN_ASSISTANT = self.ASSISTANT_ID
        self.TOKEN_TOOL_CALL = self.TOOL_CALL_ID
        self.TOKEN_TOOL_RESULT = self.TOOL_RESULT_ID
        self.TOKEN_END = self.END_ID

    # -- API ---------------------------------------------------------------
    @property
    def n_base(self) -> int:
        return N_BASE

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def special_ids(self):
        return frozenset(SPECIAL_STRINGS)

    @property
    def is_sentencepiece(self) -> bool:
        return not isinstance(self._sp, _Gpt2Fallback)

    def encode(self, text: str) -> List[int]:
        if isinstance(self._sp, _Gpt2Fallback):
            ids = self._sp.encode(text)
        else:
            ids = self._sp.encode(_map_in(text), out_type=int)
        return [int(i) for i in ids if 0 <= int(i) < N_BASE]

    def encode_batch(self, texts: Sequence[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids, skip_special: bool = False) -> str:
        parts: List[str] = []
        cur: List[int] = []
        for i in ids:
            i = int(i)
            if i >= N_BASE:
                if cur:
                    parts.append(self._sp.decode(cur))
                    cur = []
                if not skip_special:
                    parts.append(SPECIAL_STRINGS.get(i, ""))
            elif i >= 0:
                cur.append(i)
        if cur:
            parts.append(self._sp.decode(cur))
        return _map_out("".join(parts))

    # -- pickling -----------------------------------------------------------
    def __getstate__(self):
        state = {"path": str(self.model_file) if self.model_file is not None else None}
        if isinstance(self._sp, _Gpt2Fallback):
            state["kind"] = "gpt2"
            state["proto"] = None
            return state
        try:
            state["proto"] = self._sp.serialized_model_proto()
        except Exception:  # pragma: no cover - very old sentencepiece
            state["proto"] = None
        return state

    def __setstate__(self, state):
        if state.get("kind") == "gpt2":
            self._sp = _Gpt2Fallback()
        else:
            proto = state.get("proto")
            if proto:
                sp = spm.SentencePieceProcessor()
                sp.load_from_serialized_proto(proto)
                self._sp = sp
            else:
                path = state.get("path") or _find_model_path()
                self._sp = _load_sp(Path(path))
            self.model_file = Path(state["path"]) if state.get("path") else None
        self.TOKEN_ENDOFTEXT = self.EOS_ID
        self.TOKEN_SYSTEM = self.SYSTEM_ID
        self.TOKEN_USER = self.USER_ID
        self.TOKEN_ASSISTANT = self.ASSISTANT_ID
        self.TOKEN_TOOL_CALL = self.TOOL_CALL_ID
        self.TOKEN_TOOL_RESULT = self.TOOL_RESULT_ID
        self.TOKEN_END = self.END_ID


def _load_sp(path: Path):
    try:
        return spm.SentencePieceProcessor(model_file=str(path))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"cannot load Kenchi tokenizer model from {path}: {e}") from e


def _find_model_path() -> Path:
    env = os.environ.get("KENCHI_TOKENIZER")
    if env:
        return Path(env)
    cand = TOKDIR / "kenchi_sp.model"
    if cand.is_file():
        return cand
    if ASSET_PATH.is_file():
        return ASSET_PATH
    raise FileNotFoundError(
        "no Kenchi tokenizer model found (set KENCHI_TOKENIZER or build one "
        "with scripts/train_tokenizer.py; expected "
        f"{TOKDIR / 'kenchi_sp.model'} or {ASSET_PATH})")


def load_tokenizer(model_path: Optional[str] = None) -> KenchiTokenizer:
    """Canonical Kenchi tokenizer.

    Model lookup order: ``model_path`` when given, ``$KENCHI_TOKENIZER``,
    ``~/kenchi_data/tokenizer/kenchi_sp.model``, then the packaged copy
    ``src/kenchi/assets/kenchi_sp.model``.
    """
    if model_path is not None:
        return KenchiTokenizer(Path(model_path))
    return KenchiTokenizer(_find_model_path())


def tokenizer_file_sha256(model_path: Optional[str] = None) -> Optional[str]:
    """sha256 of the tokenizer model file found by the same lookup as
    ``load_tokenizer`` (``None`` when no model file exists)."""
    try:
        p = Path(model_path) if model_path else _find_model_path()
    except FileNotFoundError:
        return None
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
