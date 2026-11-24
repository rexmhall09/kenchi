#!/usr/bin/env python3
"""Kenchi chat runner (PyTorch).

    python run.py --weights model.safetensors --config config.json \
        --question "Who wrote Frankenstein?"

Without --question it is a chat in the terminal.  When the model writes a search call this script runs the search, gives the model the first
5 results and the model answers with [n] citations, or says it could not find a reliable source.

  --search          wikipedia (default, needs the internet), none, or your own function
                    (file.py:function, see search_tools.py)
  --no-self-check   turn off the self-check: a small head on the model's last layer that scores each
                    answer written from search results; under its threshold the answer is replaced
                    by "I could not find a reliable source for that."
  --history         compact (default for released weights: earlier questions and answers),
                    last (each question alone), or all (experimental full tool traffic).
                    `/new` starts a new conversation.
  --verbose         print the search results the model saw, the evidence line and its check, and the
                    self-check score

To use the model from another program (an OpenAI-style client, LangChain, smolagents, ...) run
serve.py instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kenchi.assistant import HISTORY_MODES, Assistant, default_history  # noqa: E402
from kenchi.io import load_model, read_metadata  # noqa: E402
from search_tools import load_search  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="Kenchi chat runner.")
    ap.add_argument("--weights", default="model.safetensors")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--question", default=None, help="ask one question and exit")
    ap.add_argument("--search", default="wikipedia",
                    help="wikipedia (default), none, or file.py:function / module.function")
    ap.add_argument("--system", default=None, help="a system message (default: none; the model knows who it is)")
    ap.add_argument("--no-self-check", action="store_true", help="do not let the self-check head replace answers")
    ap.add_argument("--max-tool-calls", type=int, default=1, help="searches the model may run per question")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (default: PyTorch's choice)")
    ap.add_argument("--history", nargs="?", const="all", default=None, choices=list(HISTORY_MODES),
                    help="what the model sees of the chat.  compact (the default the released weights ask for in "
                         "config.json): the earlier questions and answers as plain text, without their searches.  "
                         "last: every question is asked alone.  all (also a bare --history): every earlier turn "
                         "with its searches; the model was not trained on that")
    ap.add_argument("--no-evidence-check", action="store_true", help=argparse.SUPPRESS)   # for weights that write an evidence line; the released ones do not
    ap.add_argument("--show-thinking", action="store_true", help=argparse.SUPPRESS)   # for weights that write an evidence line; the released ones do not
    ap.add_argument("--verbose", action="store_true",
                    help="print search results, the self-check score and timings")
    ap.add_argument("--show-results", action="store_true", help=argparse.SUPPRESS)      # the old name of --verbose
    args = ap.parse_args(argv)
    verbose = args.verbose or args.show_results
    show_thinking = args.show_thinking or verbose

    for f in (args.weights, args.config):
        if not Path(f).expanduser().is_file():
            print(f"run.py: {f} not found. Pass --weights model.safetensors --config config.json "
                  "(both are in the Hugging Face folder of the release).")
            return 1
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    model, _ = load_model(args.weights, args.config, device="cpu")
    bot = Assistant(model, read_metadata(args.weights), self_check=not args.no_self_check,
                    max_new_tokens=args.max_new_tokens, history=args.history or default_history(args.config),
                    evidence_check=not args.no_evidence_check)
    backend = load_search(args.search)
    search = (lambda q: backend.search(q)) if backend else None
    system = [{"role": "system", "content": args.system}] if args.system else []
    keep = bot.history != "last"    # "compact" and "all": the chat keeps the conversation
    history = list(system)          # grows only then; in "last" mode every question is asked alone

    def on_event(kind, data):
        if kind == "tool_call":
            print(f"[search] {json.dumps(data['arguments'], ensure_ascii=False)}", flush=True)
        elif kind == "forced_search" and verbose:
            print("[the model answered with citations before searching; searching first]", flush=True)
        elif kind == "tool_result" and verbose:
            print(data, flush=True)
        elif kind == "thinking" and show_thinking:
            print(f"thinking: {data}", flush=True)
        elif kind == "evidence" and verbose:
            print("[evidence] " + ("the line holds" if data["ok"] else "the line does not hold"
                                   + (f"; answer replaced, the model had written: {data['replaced']!r}"
                                      if "replaced" in data else "; answer kept (--no-evidence-check)")), flush=True)
        elif kind == "self_check" and verbose:
            print(f"[self-check] margin {data['margin']:+.1f} (kept when 0 or more): "
                  + ("kept" if data["passed"] else f"replaced; the model had written: {data['replaced']!r}"), flush=True)

    def ask(q):
        t0 = time.time()
        if not keep:
            del history[len(system):]
        history.append({"role": "user", "content": q})
        text, new = bot.ask(history, search=search, max_tool_calls=args.max_tool_calls, on_event=on_event)
        history.extend(new)
        print(text)
        if verbose:
            print(f"[{time.time() - t0:.1f} s]")

    if args.question:
        ask(args.question)
        return 0
    print({"last": "History: last. Each question is asked alone: Kenchi does not remember the one before.",
           "compact": "History: compact. Kenchi sees the earlier questions and answers of this chat, without "
                      "their searches (/new starts over).",
           "all": "History: all. Experimental: Kenchi sees every earlier turn as it was (/new starts over). "
                  "It answers worse this way."}[bot.history])
    while True:
        try:
            q = input("you: ").strip()
        except EOFError:
            return 0
        if q == "/new":
            del history[len(system):]
            continue
        if q:
            ask(q)


if __name__ == "__main__":
    raise SystemExit(main())
