"""Does a reply say "I don't know / I could not find it"?  One list of patterns for the runners, the
self-check, the evaluations and the training scripts."""
import re

_ABSTAIN_RES = [
    re.compile(r"i\s+don'?t\s+know", re.I),
    re.compile(r"i\s+do\s+not\s+know", re.I),
    re.compile(r"i\s+couldn'?t\s+find", re.I),
    # added 2026-11-19 after the runs: "I could not find/confirm" (one of the three
    # abstention wordings in the training set) was NOT matched, so the guard, the
    # held-out check and the RL reward counted those abstentions as wrong answers
    re.compile(r"i\s+could\s+not\s+(find|confirm)", re.I),
    re.compile(r"i\s+cannot\s+find", re.I),
    re.compile(r"i\s+can'?t\s+find", re.I),
    re.compile(r"i\s+couldn'?t\s+confirm", re.I),
    re.compile(r"don'?t\s+have\s+a\s+reliable", re.I),
    re.compile(r"not\s+a\s+reliable", re.I),
    re.compile(r"no\s+reliable", re.I),
    re.compile(r"may\s+not\s+exist", re.I),
    re.compile(r"i'?m\s+not\s+(sure|certain)", re.I),
    re.compile(r"do\s+not\s+know", re.I),
]


def looks_abstained(text) -> bool:
    t = (text or "").strip().replace("\u2019", "'")
    return bool(t) and any(rx.search(t) for rx in _ABSTAIN_RES)
