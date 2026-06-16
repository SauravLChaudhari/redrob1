#!/usr/bin/env python3
"""
precompute_embeddings.py — OPTIONAL, run once, OFFLINE.

Produces the preferred semantic backend for rank.py:
    data/candidate_embeddings.npy   (N x 384 float32)
    data/jd_embedding.npy           (384,)

This step MAY use a GPU and MAY exceed the 5-minute window — the compute
constraints in the spec apply only to the *ranking step* (rank.py), not to
pre-computation. The produced .npy artifacts are committed to the repo so that
rank.py loads them with no model and no network (Stage-3 safe).

If you skip this step, rank.py automatically falls back to an in-process
TF-IDF similarity, which is fully offline and reproducible but has a lower
quality ceiling than embeddings.

Usage:
    pip install sentence-transformers
    python precompute_embeddings.py --candidates ./candidates.jsonl

Model: BAAI/bge-small-en-v1.5 (384-dim, CPU-friendly, strong retrieval model).
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, "src")
from redrob_ranker import load, jd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    texts, n = [], 0
    for c in load.stream_candidates(args.candidates):
        texts.append(load.profile_text(c))
        n += 1
    print(f"encoding {n} candidate profiles...")

    emb = model.encode(
        texts, batch_size=args.batch, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype("float32")

    jd_emb = model.encode(
        [jd.JD_TEXT], convert_to_numpy=True, normalize_embeddings=True
    )[0].astype("float32")

    os.makedirs("data", exist_ok=True)
    np.save("data/candidate_embeddings.npy", emb)
    np.save("data/jd_embedding.npy", jd_emb)
    print("wrote data/candidate_embeddings.npy and data/jd_embedding.npy")


if __name__ == "__main__":
    main()
