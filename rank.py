#!/usr/bin/env python3
"""
rank.py — Produce the top-100 submission CSV from candidates.jsonl.

Single reproduce command (Stage-3):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

CPU-only, no network. Uses precomputed embeddings (data/*.npy) if present,
otherwise an in-process TF-IDF fallback. Designed to finish well within the
5-minute / 16 GB budget for 100K candidates.
"""

import argparse
import csv
import sys
import time

sys.path.insert(0, "src")
from redrob_ranker import load, semantic, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()

    # Pass 1: stream candidates, hold the (small) fields we need.
    cands = []
    texts = []
    for c in load.stream_candidates(args.candidates):
        cands.append(c)
        texts.append(load.profile_text(c))
    print(f"[load] {len(cands)} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    # Semantic similarities (precomputed embeddings or TF-IDF fallback).
    t1 = time.time()
    sims = semantic.similarities(texts, base_dir=".")
    print(f"[semantic] {time.time()-t1:.1f}s", file=sys.stderr)

    # Score everyone.
    t2 = time.time()
    rows = []
    for c, sim in zip(cands, sims):
        s, reason, _ = score.score_candidate(c, float(sim))
        # Round to the 4 dp we actually emit, so sort order and printed score
        # stay consistent and the tie-break rule is never violated.
        rows.append((round(s, 4), c["candidate_id"], reason))
    print(f"[score] {time.time()-t2:.1f}s", file=sys.stderr)

    # Rank: score desc, tie-break candidate_id asc (matches validator rule).
    rows.sort(key=lambda r: (-r[0], r[1]))
    top = rows[: args.topk]

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (s, cid, reason) in enumerate(top, start=1):
            w.writerow([cid, i, f"{s:.4f}", reason])

    print(f"[done] wrote {args.out} ({len(top)} rows) in "
          f"{time.time()-t0:.1f}s total", file=sys.stderr)


if __name__ == "__main__":
    main()
