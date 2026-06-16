"""
semantic.py — JD <-> profile semantic similarity.

Catches the candidate the JD explicitly cares about: "A Tier 5 candidate may
not use the words 'RAG' or 'Pinecone' ... but if their career history shows
they built a recommendation system at a product company, they're a fit."

Two backends, both CPU-only and network-free at ranking time (Stage-3 safe):

  1. PRECOMPUTED EMBEDDINGS (preferred). Run precompute_embeddings.py once,
     offline, to write data/candidate_embeddings.npy + data/jd_embedding.npy
     using a small sentence-transformer (bge-small-en-v1.5). rank.py then just
     loads the .npy arrays — no model, no network.
  2. TF-IDF FALLBACK. If the .npy files are absent, we fit a TF-IDF vectorizer
     over the corpus in-process (fast, deterministic, offline). Lower ceiling
     than embeddings but fully reproducible anywhere.

The interface returns a similarity in [0, 1] per candidate, aligned to the
order in which candidates were streamed.
"""

import os
import numpy as np
from typing import List
from . import jd

EMB_CANDIDATES = "data/candidate_embeddings.npy"
EMB_JD = "data/jd_embedding.npy"


def _cosine_to_jd(mat: np.ndarray, jd_vec: np.ndarray) -> np.ndarray:
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    jd_vec = jd_vec / (np.linalg.norm(jd_vec) + 1e-9)
    sims = mat @ jd_vec
    return (sims + 1.0) / 2.0  # map [-1,1] -> [0,1]


def similarities(profile_texts: List[str], base_dir: str = ".") -> np.ndarray:
    cand_path = os.path.join(base_dir, EMB_CANDIDATES)
    jd_path = os.path.join(base_dir, EMB_JD)

    if os.path.exists(cand_path) and os.path.exists(jd_path):
        mat = np.load(cand_path)
        jd_vec = np.load(jd_path)
        if mat.shape[0] == len(profile_texts):
            return _cosine_to_jd(mat, jd_vec)
        # Shape mismatch -> fall through to TF-IDF rather than misalign.

    # TF-IDF fallback.
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(
        max_features=40000, ngram_range=(1, 2), sublinear_tf=True,
        stop_words="english", min_df=3,
    )
    corpus = profile_texts + [jd.JD_TEXT]
    tfidf = vec.fit_transform(corpus)
    jd_row = tfidf[-1]
    cand_rows = tfidf[:-1]
    sims = (cand_rows @ jd_row.T).toarray().ravel()
    # Normalize to [0,1] by the observed max for stable blending.
    mx = sims.max() if sims.size else 1.0
    return sims / (mx + 1e-9)
