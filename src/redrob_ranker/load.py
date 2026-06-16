"""
load.py — Stream candidates from the JSONL file.

We stream line-by-line rather than json.load the whole 487 MB file, to stay
well inside the 16 GB memory budget. We also precompute a couple of cheap
derived fields used by several scorers (a flattened skills view and a
profile-text blob for the semantic layer).
"""

import json
from typing import Iterator, Dict, Any


def stream_candidates(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def profile_text(c: Dict[str, Any]) -> str:
    """Concatenate the free-text fields used for semantic similarity."""
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for role in c.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    parts.append(" ".join(s.get("name", "") for s in c.get("skills", [])))
    return " ".join(x for x in parts if x)


def all_titles(c: Dict[str, Any]) -> list:
    """Current title + every career-history title, lowercased."""
    p = c.get("profile", {})
    out = [p.get("current_title", "")]
    out += [r.get("title", "") for r in c.get("career_history", [])]
    out += [p.get("headline", "")]
    return [t.lower() for t in out if t]
