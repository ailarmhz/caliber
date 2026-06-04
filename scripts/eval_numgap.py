#!/usr/bin/env python3
"""Evaluate a checkpoint on NumGap.

For each record, with s_p = cos(emb(anchor), emb(perturbation)) and
s_d = cos(emb(anchor), emb(distractor)), a record is correct iff s_p < s_d
(the numeric edit is a larger semantic break than an unrelated topical neighbor).

    NumGap-D = mean correct           NumGap-M = mean (s_d - s_p)

Overall and per-category scores are written to CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def load_records(path: str, split: str = "test") -> List[Dict]:
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            r = json.loads(ln)
            if split and r.get("split") != split: continue
            out.append(r)
    return out


@torch.no_grad()
def encode(model, tok, texts: List[str], max_length=512, batch_size=32) -> np.ndarray:
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        b = texts[i:i+batch_size]
        inp = tok(b, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        h = model(**inp).last_hidden_state
        mask = inp["attention_mask"].unsqueeze(-1).to(h.dtype)
        emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        emb = F.normalize(emb, p=2, dim=-1)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1024), dtype="float32")


def evaluate(args):
    records = load_records(args.numgap, args.split)
    print(f"[numgap_eval] {len(records)} records (split={args.split})")
    if not records:
        print("WARN: no records; check --split"); return

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16, trust_remote_code=False)
    model.eval()
    if torch.cuda.is_available(): model = model.to("cuda")

    anchors = [r["anchor"] for r in records]
    perts = [r["perturbation"] for r in records]
    dists = [r["distractor"] for r in records]
    cats = [r["category"] for r in records]

    print("[numgap_eval] encoding…")
    a = encode(model, tok, anchors, args.max_length, args.batch_size)
    p = encode(model, tok, perts, args.max_length, args.batch_size)
    d = encode(model, tok, dists, args.max_length, args.batch_size)

    s_p = (a * p).sum(axis=-1)
    s_d = (a * d).sum(axis=-1)
    correct = (s_p < s_d).astype("float32")

    overall_d = float(correct.mean())
    overall_m = float((s_d - s_p).mean())

    by_cat: Dict[str, list] = defaultdict(list)
    by_cat_m: Dict[str, list] = defaultdict(list)
    for c, ok, mm in zip(cats, correct, s_d - s_p):
        by_cat[c].append(float(ok)); by_cat_m[c].append(float(mm))

    rows = [{"category": "overall",
             "n": len(records),
             "numgap_d": overall_d,
             "numgap_m": overall_m}]
    for c in sorted(by_cat):
        rows.append({"category": c, "n": len(by_cat[c]),
                     "numgap_d": float(np.mean(by_cat[c])),
                     "numgap_m": float(np.mean(by_cat_m[c]))})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "category", "n", "numgap_d", "numgap_m"])
        w.writeheader()
        for r in rows: w.writerow({"model": os.path.basename(os.path.normpath(args.model)), **r})
    print(f"[numgap_eval] wrote {args.out}")
    for r in rows:
        print(f"  {r['category']:10s} n={r['n']:5d}  D={r['numgap_d']:.4f}  M={r['numgap_m']:+.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--numgap", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="test", choices=["dev", "test", ""])
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out", default="numgap_metrics.csv")
    args = p.parse_args()
    evaluate(args)
