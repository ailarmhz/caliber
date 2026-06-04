#!/usr/bin/env python3
"""Build the NumGap evaluation set.

Reads a corpus parquet (text_id, text, source) and writes (anchor, perturbation,
distractor) triples as JSONL, stratified by perturbation category and split into
dev/test. A companion ``*_card.json`` records counts, sources, and length stats.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# Make repo root importable
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from caliber.perturb import Perturber, NUM_RE


CATEGORY_TARGETS = {
    "magnitude": 0.25,
    "polarity":  0.25,
    "period":    0.20,
    "unit":      0.15,
    "currency":  0.15,
}


def _eligible(text: str) -> bool:
    if not isinstance(text, str): return False
    n = len(text)
    if n < 200 or n > 1200: return False
    if not re.search(r"[.!?]", text): return False
    return len(NUM_RE.findall(text)) >= 2


def _bm25_distractor_indices(corpus_texts: List[str], anchor_idx: int,
                             top_k: int = 100) -> List[int]:
    """Cheap BM25 via rank_bm25 if available; otherwise TF cosine fallback."""
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in corpus_texts]
        bm25 = BM25Okapi(tokenized)
        q = corpus_texts[anchor_idx].lower().split()
        scores = bm25.get_scores(q)
    except ImportError:
        # TF-only fallback (less ideal but no extra deps)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", max_features=20000)
        X = vec.fit_transform(corpus_texts)
        scores = (X[anchor_idx] @ X.T).toarray().ravel()
    order = np.argsort(-scores)
    return [int(i) for i in order[:top_k] if i != anchor_idx]


def _numeric_jaccard(a: str, b: str) -> float:
    A = set(m.group() for m in NUM_RE.finditer(a))
    B = set(m.group() for m in NUM_RE.finditer(b))
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))


def _len_diff(a: str, b: str) -> int:
    return abs(len(a) - len(b))


def build(args) -> None:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.corpus)
    if "text" not in df.columns:
        raise ValueError(f"corpus parquet missing 'text' column; got {list(df.columns)}")
    if "source" not in df.columns:
        df["source"] = "unknown"

    print(f"[numgap] loaded {len(df):,} rows from {args.corpus}")
    df = df.reset_index(drop=True)

    # 1) Find eligible anchor passages
    elig_mask = df["text"].apply(_eligible)
    eligible_idx = list(df.index[elig_mask])
    print(f"[numgap] {len(eligible_idx):,} eligible passages")
    if len(eligible_idx) < args.n_test + args.n_dev:
        raise RuntimeError(f"not enough eligible passages: {len(eligible_idx)}")

    # 2) Generate perturbations, stratified by category
    n_target = args.n_test + args.n_dev
    target_per_cat = {c: int(round(p * n_target)) for c, p in CATEGORY_TARGETS.items()}
    print(f"[numgap] target per category: {target_per_cat}")

    pert = Perturber(seed=args.seed)
    candidates_by_cat: Dict[str, List[Dict]] = {c: [] for c in target_per_cat}

    rng.shuffle(eligible_idx)
    for idx in eligible_idx:
        if all(len(v) >= target_per_cat[c] for c, v in candidates_by_cat.items()):
            break
        text = df.at[idx, "text"]
        out = pert.perturb(text, key=str(df.at[idx, "text_id"]) if "text_id" in df.columns else None)
        if out is None: continue
        if candidates_by_cat[out.category] and \
                len(candidates_by_cat[out.category]) >= target_per_cat[out.category]:
            continue
        # Edit-distance hard filter
        if _len_diff(text, out.text) > 30: continue
        if _len_diff(text, out.text) < 1: continue
        candidates_by_cat[out.category].append({
            "anchor_idx": int(idx),
            "anchor": text,
            "perturbation": out.text,
            "category": out.category,
            "source": df.at[idx, "source"] if "source" in df.columns else "unknown",
        })

    n_made = sum(len(v) for v in candidates_by_cat.values())
    print(f"[numgap] candidates generated: {n_made}")
    for c, v in candidates_by_cat.items():
        print(f"  {c}: {len(v)}")

    # 3) Optional soft filter via MiniLM cosine similarity
    if args.filter_model:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(args.filter_model)
        for c in candidates_by_cat:
            kept = []
            for r in candidates_by_cat[c]:
                emb = st.encode([r["anchor"], r["perturbation"]],
                                normalize_embeddings=True, convert_to_numpy=True)
                cos = float(emb[0] @ emb[1])
                if cos >= 0.85:
                    kept.append(r)
            print(f"[numgap] filter_model kept {len(kept)}/{len(candidates_by_cat[c])} for {c}")
            candidates_by_cat[c] = kept

    # 4) Distractor sampling via BM25 over the eligible corpus
    print("[numgap] sampling distractors via BM25 …")
    eligible_texts = df.loc[eligible_idx, "text"].tolist()
    eligible_pos = {idx: i for i, idx in enumerate(eligible_idx)}

    # Build BM25 once on the eligible corpus
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [t.lower().split() for t in eligible_texts]
        bm25 = BM25Okapi(tokenized)
        use_bm25 = True
    except ImportError:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(stop_words="english", max_features=20000)
        X = vec.fit_transform(eligible_texts)
        use_bm25 = False

    final_records: List[Dict] = []
    rec_id = 0
    for c, recs in candidates_by_cat.items():
        for r in recs:
            anchor = r["anchor"]; pos = eligible_pos.get(r["anchor_idx"])
            if pos is None: continue
            if use_bm25:
                scores = bm25.get_scores(anchor.lower().split())
            else:
                scores = (X[pos] @ X.T).toarray().ravel()
            order = np.argsort(-scores)

            distractor = None
            for j in order[:200]:
                if j == pos: continue
                cand = eligible_texts[j]
                if _numeric_jaccard(anchor, cand) > 0.5: continue
                if _len_diff(anchor, cand) > 200: continue
                distractor = cand; break

            if distractor is None: continue

            final_records.append({
                "id": f"numgap_{rec_id:06d}",
                "category": c,
                "anchor": anchor,
                "perturbation": r["perturbation"],
                "distractor": distractor,
                "source": r["source"],
                "split": None,  # filled below
            })
            rec_id += 1

    print(f"[numgap] {len(final_records)} records after distractor sampling")

    # 5) Stratified dev/test split
    rng.shuffle(final_records)
    by_cat = {}
    for r in final_records:
        by_cat.setdefault(r["category"], []).append(r)
    out_recs = []
    for c, lst in by_cat.items():
        rng.shuffle(lst)
        n_dev_c = int(round(args.n_dev * CATEGORY_TARGETS[c]))
        for r in lst[:n_dev_c]: r["split"] = "dev";  out_recs.append(r)
        for r in lst[n_dev_c:]: r["split"] = "test"; out_recs.append(r)

    rng.shuffle(out_recs)

    # 6) Write
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 7) Card
    cnt = Counter((r["split"], r["category"]) for r in out_recs)
    card = {
        "size": len(out_recs),
        "by_split_category": {f"{s}:{c}": v for (s, c), v in cnt.items()},
        "sources": Counter(r["source"] for r in out_recs),
        "median_anchor_len": int(np.median([len(r["anchor"]) for r in out_recs])),
        "median_pert_len": int(np.median([len(r["perturbation"]) for r in out_recs])),
        "median_dist_len": int(np.median([len(r["distractor"]) for r in out_recs])),
    }
    card_path = args.out.replace(".jsonl", "_card.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2, default=str)
    print(f"[numgap] wrote {len(out_recs)} records to {args.out}")
    print(f"[numgap] card -> {card_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="parquet with text_id, text, source")
    p.add_argument("--out", required=True, help="output jsonl path")
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--n_dev", type=int, default=500)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--filter_model", default=None,
                   help="optional st model for cosine soft-filter; e.g. sentence-transformers/all-MiniLM-L6-v2")
    args = p.parse_args()
    build(args)
