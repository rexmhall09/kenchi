#!/usr/bin/env python3
"""OpenAI-compatible HTTP server for Kenchi (scripts/release_templates/serve.py).

Serves ``POST /v1/chat/completions``, ``GET /v1/models`` and ``GET /health`` so
stock harnesses (the OpenAI SDK, LangChain, smolagents, LiteLLM, ...) can drive
Kenchi.  Prompts go through the canonical renderer ``kenchi.format_v2.render``
and answers through ``kenchi.format_v2.parse_output``, so the wire format is the
Kenchi v2 format the model was trained on.

Backends (--backend):

* ``torch`` (default): the PyTorch model on CPU.
* ``mlx``: the MLX model (Apple Silicon), same behaviour.
* ``fake``: a scripted generator for testing a client without loading the model.

Protocol notes:

* ``tools`` are shown to the model as they are (name, description, JSON schema), whatever they
  are called.  ``tool_choice`` ``"none"`` hides them; ``"required"`` and a named function are
  treated as ``"auto"`` (the model is not forced).
* Tool-call ids are made up per response (``call_`` + random hex); the model format has no ids, so
  ``tool`` messages are matched to calls by order.
* A prompt longer than the context (2,048 tokens minus max new tokens) is rejected with HTTP 400
  and an OpenAI-style error that says how many tokens it has; nothing is cut silently.
* ``stream: true`` is answered with SSE chunks, but the text is generated first and sent at once.
* The self-check (see run.py) is on unless ``--no-self-check``.  With ``--builtin-search wikipedia``
  a request WITHOUT tools is answered by letting the model search by itself.
* ``--history last|compact|all``: what the model is shown of the conversation in a request (see
  run.py).  ``last`` is the default (the environment variable KENCHI_HISTORY changes it).
* The evidence line the model writes before an answer from tool results (``[2] says: "..."`` or
  ``No result states this.``) is not part of ``content``: it is returned as
  ``message.reasoning_content`` (streaming: one ``reasoning_content`` delta before the content).
  The line is checked against the tool result (off with ``--no-evidence-check``); the verdict is
  under the top-level ``kenchi`` key of the response (``evidence``), next to ``self_check``.

Usage:

    python serve.py --weights model.safetensors --config config.json --port 8000
    python serve.py --backend mlx --weights model.safetensors --config config.json
    python serve.py --backend fake --port 8123 --log-requests /tmp/req.log
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kenchi import format_v2  # noqa: E402
from kenchi.assistant import default_history as _default_history  # noqa: E402

MODEL_ID = "kenchi"
MAX_CONTEXT_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 256
_LOCK = threading.Lock()
_LOG_PATH: str | None = None


# ------------------------------------------------------------------ fake generator

def _fake_generate(messages, tools):
    """Scripted generator for plumbing tests; returns token ids through parse_output."""
    from kenchi import tokenizer as kt

    last = messages[-1] if messages else {}
    role = last.get("role")
    if role == "tool":
        text = "Based on the results: " + format_v2.content_text(last.get("content"))[:60]
        return kt.encode_text(text) + [format_v2.END]
    if role == "user":
        user_text = format_v2.content_text(last.get("content"))
        if tools:
            spec = format_v2.tool_spec(tools[0])
            params = spec.get("parameters") or {}
            required = params.get("required") or []
            args = {p: user_text for p in required}
            call = {"name": spec.get("name", ""), "arguments": args}
            return [format_v2.TOOL_CALL] + kt.encode_text(format_v2.dumps(call)) + [format_v2.END]
        return kt.encode_text(user_text) + [format_v2.END]
    return [format_v2.END]


# ------------------------------------------------------------------ backend

class _Backend:
    """Turns OpenAI-style messages + tools into one assistant message (see kenchi/assistant.py)."""

    def __init__(self, kind, weights=None, config=None, self_check=True, search=None, threads=0, history=None,
                 evidence_check=True):
        from kenchi.assistant import history_mode
        self.history = history_mode(history)             # "last", "compact" or "all"
        self.kind, self.weights, self.config = kind, weights, config
        self.self_check, self.search = self_check, search
        self.bot = None
        if kind == "torch":
            import torch
            if threads:
                torch.set_num_threads(threads)
            from kenchi.assistant import Assistant
            from kenchi.io import load_model, read_metadata
            model, _ = load_model(weights, config, device="cpu")
            self.bot = Assistant(model, read_metadata(weights), self_check=self_check, history=self.history,
                                 evidence_check=evidence_check)
        elif kind == "mlx":
            from run_mlx import MlxAssistant
            self.bot = MlxAssistant(weights, config, self_check=self_check, history=self.history,
                                    evidence_check=evidence_check)

    def count(self, messages, tools):
        return len(format_v2.render(messages, tools))

    def complete(self, messages, tools, max_new_tokens, temperature=0.0):
        """-> assistant message dict (format_v2.parse_output keys + prompt_tokens, completion_tokens)."""
        if self.kind == "fake":
            ids = _fake_generate(messages, tools)
            msg = format_v2.parse_output(ids)
            msg["prompt_tokens"], msg["completion_tokens"] = self.count(messages, tools), len(ids)
            return msg
        if not tools and self.search is not None:
            # a plain chat client that sends no tools: the server runs the search itself
            seen = {}
            text, new = self.bot.ask(messages, search=self.search, on_event=seen.__setitem__)
            out = {"role": "assistant", "content": text, "finish_reason": "stop",
                   "prompt_tokens": self.count(list(messages) + new[:-1], [format_v2.WEB_SEARCH_TOOL]),
                   "completion_tokens": len(format_v2.render([new[-1]], None, add_generation_prompt=False))}
            if new[-1].get("reasoning_content"):         # the evidence line of the final answer
                out["reasoning_content"], out["evidence"] = new[-1]["reasoning_content"], seen.get("evidence")
            return out
        return self.bot.complete(messages, tools, max_new_tokens=max_new_tokens, temperature=temperature)


# ------------------------------------------------------------------ response building

def _error(message, err_type, param=None, code=None):
    return {"error": {"message": message, "type": err_type,
                      "param": param, "code": code}}


def _tool_calls_of(msg):
    if not msg.get("tool_calls"):
        return None
    out = []
    for tc in msg["tool_calls"]:
        out.append({"id": "call_" + secrets.token_hex(8), "type": "function",
                    "function": {"name": tc["function"]["name"],
                                 "arguments": tc["function"]["arguments"]}})
    return out


def _completion(msg, prompt_tokens, completion_tokens, model_id):
    tool_calls = _tool_calls_of(msg)
    finish = msg.get("finish_reason", "stop")
    if msg.get("invalid_tool_call") is not None:
        finish = "stop"
    message = {"role": "assistant", "content": msg.get("content"), "refusal": None, "tool_calls": tool_calls}
    if msg.get("reasoning_content"):                    # the evidence line (see the protocol notes)
        message["reasoning_content"] = msg["reasoning_content"]
    resp = {
        "id": "chatcmpl-" + secrets.token_hex(12),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }
    extra = {k: msg[k] for k in _KENCHI_KEYS if msg.get(k) is not None}
    if extra:
        resp["kenchi"] = extra                          # not part of the OpenAI format; clients ignore it
    return resp


_KENCHI_KEYS = ("self_check", "evidence", "forced_tool_call", "citations_removed", "abstention_reworded")


def _stream_deltas(msg, tool_calls):
    """The deltas of a streamed reply, in order: the evidence line (`reasoning_content`) if there is
    one, then the content and the tool calls.  The first delta carries the role."""
    delta: dict = {}
    if msg.get("content"):
        delta["content"] = msg["content"]
    if tool_calls:
        delta["tool_calls"] = [{
            "index": i, "id": tc["id"], "type": "function",
            "function": {"name": tc["function"]["name"],
                         "arguments": tc["function"]["arguments"]}}
            for i, tc in enumerate(tool_calls)]
    deltas = [{"reasoning_content": msg["reasoning_content"]}, delta] if msg.get("reasoning_content") else [delta]
    deltas[0] = {"role": "assistant", **deltas[0]}
    return deltas


# ------------------------------------------------------------------ HTTP server

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- helpers
    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_chunk(self, cid, created, model_id, delta, finish_reason):
        obj = {"id": cid, "object": "chat.completion.chunk", "created": created,
               "model": model_id,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        return f"data: {json.dumps(obj)}\n\n"

    # -- routes
    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "created": int(time.time()),
                 "owned_by": "kenchi"}]})
        else:
            self._json(404, _error(f"The requested path was not found: {self.path}",
                                    "invalid_request_error"))

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, _error(f"The requested path was not found: {self.path}",
                                   "invalid_request_error"))
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._json(400, _error("The request body is not valid JSON.", "invalid_request_error"))
            return
        if not isinstance(req, dict) or not req.get("messages"):
            self._json(400, _error("messages is a required field.", "invalid_request_error",
                                   param="messages"))
            return
        if _LOG_PATH:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": time.time(), "path": self.path, "body": req}) + "\n")
        with _LOCK:                                    # one request at a time
            self._chat_completions(req)

    def _chat_completions(self, req):
        model_id = req.get("model") or MODEL_ID
        messages = req["messages"]
        tools = req.get("tools") or None
        # tool_choice "none": the tools are not shown to the model.  "required" and a named function
        # are treated as "auto": the model decides (it is not forced).
        if req.get("tool_choice") == "none":
            tools = None
        max_tokens = req.get("max_tokens") or req.get("max_completion_tokens") or DEFAULT_MAX_NEW_TOKENS
        try:
            max_new = int(max_tokens)
        except (TypeError, ValueError):
            max_new = DEFAULT_MAX_NEW_TOKENS
        temperature = float(req.get("temperature") or 0.0)
        stream = bool(req.get("stream"))

        # what the model is shown of the conversation: "last" (the last user message and what followed
        # it), "compact" (the earlier turns as plain exchanges, cut to fit) or "all" (everything)
        from kenchi.assistant import shown_messages
        messages = shown_messages(messages, self.server.backend.history, tools, max_new)
        prompt_ids = format_v2.render(messages, tools)
        prompt_tokens = len(prompt_ids)
        limit = MAX_CONTEXT_TOKENS - max_new
        if prompt_tokens > limit:
            self._json(400, _error(
                f"This model's maximum context length is {MAX_CONTEXT_TOKENS} tokens. "
                f"Your prompt is {prompt_tokens} tokens long; with max_tokens={max_new} the "
                f"allowed maximum is {limit} tokens. Reduce the prompt or lower max_tokens.",
                "invalid_request_error", param="messages", code="context_length_exceeded"))
            return

        msg = self.server.backend.complete(messages, tools, max_new, temperature)
        prompt_tokens, completion_tokens = msg.get("prompt_tokens", prompt_tokens), msg.get("completion_tokens", 0)

        if stream:
            self._sse(msg, model_id)
            return
        self._json(200, _completion(msg, prompt_tokens, completion_tokens, model_id))

    def _sse(self, msg, model_id):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        cid = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())
        tool_calls = _tool_calls_of(msg)
        finish = msg.get("finish_reason", "stop")
        if msg.get("invalid_tool_call") is not None:
            finish = "stop"
        for delta in _stream_deltas(msg, tool_calls):
            self.wfile.write(self._sse_chunk(cid, created, model_id, delta, None).encode())
        self.wfile.write(self._sse_chunk(cid, created, model_id, {}, finish).encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, backend):
        super().__init__(addr, _Handler)
        self.backend = backend


def main(argv=None):
    global _LOG_PATH
    ap = argparse.ArgumentParser(description="Kenchi OpenAI-compatible server.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--weights", default=None,
                    help="safetensors (torch backend); default: the release weights")
    ap.add_argument("--config", default=None,
                    help="config JSON (torch/mlx backends); default: the release config")
    ap.add_argument("--backend", default="torch", choices=["torch", "mlx", "fake"])
    ap.add_argument("--log-requests", default=None,
                    help="append every POST request body (JSON lines) to this file")
    ap.add_argument("--no-self-check", action="store_true",
                    help="do not let the self-check head replace answers (see run.py)")
    ap.add_argument("--builtin-search", default="none",
                    help="for clients that send no tools: wikipedia, none (default), or file.py:function. "
                         "The server then runs the model's search itself and returns the final answer")
    ap.add_argument("--history", default=None, choices=["last", "compact", "all"],
                    help="what the model sees of a conversation.  compact (the default the released weights ask "
                         "for in config.json): the earlier user messages and final answers as plain text (at most "
                         "400 tokens of them), without their tool calls and results.  last: only the last user "
                         "message and what followed it.  all: every turn the client sent; the model was not "
                         "trained on that")
    ap.add_argument("--no-evidence-check", action="store_true", help=argparse.SUPPRESS)   # for weights that write an evidence line; the released ones do not
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for the torch backend")
    args = ap.parse_args(argv)

    _LOG_PATH = args.log_requests

    weights = args.weights or "model.safetensors"
    config = args.config or "config.json"

    if args.backend != "fake" and not (Path(weights).expanduser().is_file() and Path(config).expanduser().is_file()):
        print(f"serve.py: {weights} or {config} not found. Pass --weights model.safetensors --config config.json "
              "(both are in the Hugging Face folder of the release).", file=sys.stderr)
        return 1
    search = None
    if args.builtin_search != "none":
        from search_tools import load_search
        be = load_search(args.builtin_search)
        search = (lambda q: be.search(q)) if be else None

    backend = _Backend(args.backend, weights, config, self_check=not args.no_self_check, search=search,
                       threads=args.threads, history=args.history or _default_history(config), evidence_check=not args.no_evidence_check)
    server = _Server((args.host, args.port), backend)
    print(f"Kenchi server (backend={args.backend}, history={backend.history}) listening on "
          f"http://{server.server_address[0]}:{server.server_address[1]}/v1/chat/completions",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
