#!/usr/bin/env python3
"""Asymmetric retrieval evaluator.

Given a triplets JSONL (qid, query, pos[], neg[]), computes nDCG, Recall, MRR,
and MAP at several cutoffs in three encoding modes and writes them to CSV:
student/student, asymmetric (student query + teacher document), and
teacher/teacher (upper bound).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ---- IO -----------------------------------------------------------------

def load_triplets(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            r = json.loads(ln)
            qid = r.get("qid") or r.get("id")
            query = r.get("query") or r.get("question") or ""
            pos = r.get("pos") or [r.get("pos_doc")] if "pos_doc" in r else r.get("pos") or []
            if "pos_doc" in r and r["pos_doc"]: pos = [r["pos_doc"]]
            neg_field = r.get("neg") or r.get("neg_docs") or []
            neg = []
            for n in neg_field:
                if isinstance(n, str): neg.append(n)
                elif isinstance(n, dict) and "text" in n: neg.append(n["text"])
            rows.append({"qid": qid, "query": query, "pos": [p for p in pos if p], "neg": neg})
    return [r for r in rows if r["query"] and r["pos"]]


# ---- encoder wrapper ----------------------------------------------------

class Encoder:
    def __init__(self, model_path: str, max_length: int = 512, batch_size: int = 32):
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=False
        )
        self.model.eval()
        if torch.cuda.is_available(): self.model = self.model.to("cuda")
        self.device = next(self.model.parameters()).device
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            b = texts[i:i+self.batch_size]
            inp = self.tok(b, padding=True, truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            h = self.model(**inp).last_hidden_state
            mask = inp["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            emb = F.normalize(emb, p=2, dim=-1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1024), dtype="float32")


# ---- metrics ------------------------------------------------------------

def metrics_for_query(scores: np.ndarray, rel_idx: set, ks=(1, 5, 10, 100, 1000)) -> Dict[str, float]:
    order = np.argsort(-scores); hits = [1 if i in rel_idx else 0 for i in order]
    out = {}
    for k in ks:
        kk = min(k, len(hits))
        nrel = len(rel_idx)
        out[f"recall_at_{k}"] = sum(hits[:kk]) / nrel if nrel > 0 else 0.0
        out[f"precision_at_{k}"] = sum(hits[:kk]) / max(1, kk)
        # MRR
        mrr = 0.0
        for i in range(kk):
            if hits[i]: mrr = 1.0/(i+1); break
        out[f"mrr_at_{k}"] = mrr
        # MAP
        s = 0.0; nh = 0
        for i in range(kk):
            if hits[i]: nh += 1; s += nh/(i+1)
        out[f"map_at_{k}"] = s / nrel if nrel > 0 else 0.0
        # nDCG
        dcg = sum(1.0/math.log2(i+2) for i in range(kk) if hits[i])
        ideal = sum(1.0/math.log2(i+2) for i in range(min(nrel, kk)))
        out[f"ndcg_at_{k}"] = dcg/ideal if ideal > 0 else 0.0
    return out


# ---- main eval ----------------------------------------------------------

def evaluate(args):
    rng = random.Random(args.seed)
    rows = load_triplets(args.triplets)
    if args.num_queries < len(rows):
        rows = rng.sample(rows, args.num_queries)

    # Build a fragment: for each query, pool = pos + neg + global pad
    global_docs = []
    for r in rows: global_docs += r["pos"] + r["neg"]
    seen = set(); global_uniq = []
    for t in global_docs:
        if t not in seen: seen.add(t); global_uniq.append(t)

    queries = []; corpora = []; rel_sets = []
    for r in rows:
        pool = list(r["pos"])
        rng.shuffle(r["neg"])
        for t in r["neg"]:
            if len(pool) >= args.pool_size: break
            if t not in pool: pool.append(t)
        if len(pool) < args.pool_size:
            for t in global_uniq:
                if len(pool) >= args.pool_size: break
                if t not in pool: pool.append(t)
        rel_idx = {i for i, t in enumerate(pool) if t in set(r["pos"])}
        if not rel_idx: continue
        queries.append(r["query"]); corpora.append(pool); rel_sets.append(rel_idx)
    print(f"[eval_asym] {len(queries)} usable queries, pool size {args.pool_size}")

    out_csv = open(args.out, "w", newline="")
    writer = None

    def write_metrics(model_name: str, mode: str, q_emb: np.ndarray, doc_embs: List[np.ndarray]):
        nonlocal writer
        per = []
        for q, dE, rel in zip(q_emb, doc_embs, rel_sets):
            scores = (dE @ q).reshape(-1)
            per.append(metrics_for_query(scores, rel))
        agg = {k: float(np.mean([m[k] for m in per])) for k in per[0]}
        agg["main_score"] = agg.get("ndcg_at_10", 0.0)
        agg["accuracy"] = agg.get("recall_at_1", 0.0)
        row = {"model": model_name, "mode": mode, **agg}
        if writer is None:
            writer = csv.DictWriter(out_csv, fieldnames=list(row.keys())); writer.writeheader()
        writer.writerow(row)
        print(f"[eval_asym] {model_name:30s} {mode:8s} nDCG@10={agg['ndcg_at_10']:.4f} "
              f"R@10={agg['recall_at_10']:.4f} R@100={agg['recall_at_100']:.4f}")

    # 1) Encode docs once with each encoder (same pools per query)
    student = Encoder(args.student) if args.student else None
    teacher = Encoder(args.teacher) if args.teacher else None

    def encode_pools(enc):
        out = []
        for pool in corpora: out.append(enc.encode(pool))
        return out

    if student is not None:
        s_q = student.encode(queries); s_doc = encode_pools(student)
    if teacher is not None:
        t_q = teacher.encode(queries); t_doc = encode_pools(teacher)

    name = os.path.basename(os.path.normpath(args.student)) if args.student else "no_student"

    if student is not None: write_metrics(name, "student/student", s_q, s_doc)
    if student is not None and teacher is not None:
        write_metrics(name, "asymmetric", s_q, t_doc)
    if teacher is not None:
        write_metrics(os.path.basename(os.path.normpath(args.teacher)),
                      "teacher/teacher", t_q, t_doc)

    out_csv.close()
    print(f"[eval_asym] wrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--triplets", required=True)
    p.add_argument("--student", default=None, help="path to student checkpoint")
    p.add_argument("--teacher", default=None, help="path to teacher checkpoint")
    p.add_argument("--num_queries", type=int, default=200)
    p.add_argument("--pool_size", type=int, default=200)
    p.add_argument("--out", default="metrics_asym.csv")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    evaluate(args)
