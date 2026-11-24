"""Search for the Kenchi runners (run.py and run_mlx.py). Standard library only.

A search is any function `search(query)` that returns a list of
`{"title": ..., "url": ..., "snippet": ...}`. The runner shows the model the
first 5 results, each snippet cut to 60 words, the layout it saw most in training.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def title_first(query, results):
    """Put the page whose title IS the query at the top.

    Wikipedia's search ranks by popularity as much as by match: "Obi-Wan Kenobi" returns the
    miniseries before the character, "airplane" returns the film "Airplane!" before the aircraft.
    The model mostly answers from the first result, so the page named exactly like the query (an
    article in front and hyphens aside) goes first.  Nothing else is reordered."""
    def key(text):
        return re.sub(r"[\s\-_]+", " ", str(text).strip().lower())

    exact = key(query).rstrip(" ?.!")
    for want in (exact, re.sub(r"^(?:an?|the) ", "", exact)):       # the title as typed, then without an article
        for i, r in enumerate(results):
            if want and key(r.get("title", "")) == want:
                return [r] + list(results[:i]) + list(results[i + 1:])
    return list(results)


def wikipedia_search(query, n=5):
    """Top `n` Wikipedia pages for the query, each with the start of its intro. One request.

    Wikipedia limits how often a client may ask (HTTP 429 with a Retry-After header); this waits
    once for as long as it says (at most 40 s) and tries again."""
    params = {"action": "query", "generator": "search", "gsrsearch": str(query), "gsrlimit": int(n),
              "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": "max", "format": "json"}
    url = WIKIPEDIA_API + "?" + urllib.parse.urlencode(params)
    # Wikipedia refuses requests without a User-Agent that names the client
    req = urllib.request.Request(url, headers={"User-Agent": "kenchi-runner/2.0 (local use)"})
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 2:
                raise
            time.sleep(min(40.0, float(e.headers.get("Retry-After") or 20) + 1.0))
    pages = sorted((data.get("query", {}).get("pages", {}) or {}).values(), key=lambda p: p.get("index", 999))
    return title_first(query, [{"title": p.get("title", ""),
                                "url": "https://en.wikipedia.org/wiki/" + p.get("title", "").replace(" ", "_"),
                                "snippet": " ".join((p.get("extract") or "").split()[:80])} for p in pages[:n]])


class SearchBackend:
    """Wraps a `search(query)` function as the object the chat loop calls."""

    def __init__(self, fn):
        self.fn = fn

    def search(self, query, k=5):
        return list(self.fn(query))[:k]


def load_search(spec):
    """`--search` value -> SearchBackend, or None for no search.

    "wikipedia" (default), "none", or your own function as `file.py:function`
    or `module.function`.
    """
    spec = (spec or "wikipedia").strip()
    if spec == "none":
        return None
    if spec == "wikipedia":
        return SearchBackend(wikipedia_search)
    if ":" in spec:
        path, _, attr = spec.rpartition(":")
        s = importlib.util.spec_from_file_location(Path(path).stem, Path(path).expanduser())
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
    else:
        name, _, attr = spec.rpartition(".")
        if not name:
            raise ValueError(f"bad --search value {spec!r}: use file.py:function or module.function")
        mod = importlib.import_module(name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise ValueError(f"{spec!r} is not a function")
    return SearchBackend(fn)
