#!/usr/bin/env python3
"""Precompute teacher embeddings for Caliber training.

Reads a corpus parquet (text_id, text, source), generates one numeric
perturbation per eligible text, and writes a single parquet with columns:
text_id, text, source, has_perturb, perturb_category, perturb_text,
teacher_emb (float32 [d]), and teacher_perturb_emb (float32 [d] or null).
This parquet is the only input the trainer needs.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from caliber.perturb import Perturber  # noqa: E402


def _load_teacher(model_path: str, dtype_str: str = "bfloat16"):
    from transformers import AutoModel, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[dtype_str]
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tok


@torch.no_grad()
def _encode(texts: List[str], model, tok, max_length: int, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    out_chunks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inp = tok(batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        h = model(**inp).last_hidden_state                       # [b, t, d]
        mask = inp["attention_mask"].unsqueeze(-1).to(h.dtype)
        emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        emb = torch.nn.functional.normalize(emb, dim=-1, p=2)
        out_chunks.append(emb.float().cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def main(args):
    df = pd.read_parquet(args.corpus)
    if "text" not in df.columns:
        raise ValueError(f"corpus needs a 'text' column; got {list(df.columns)}")
    if "text_id" not in df.columns:
        df["text_id"] = [f"t{i:09d}" for i in range(len(df))]
    if "source" not in df.columns:
        df["source"] = "unknown"
    print(f"[extract] {len(df):,} rows from {args.corpus}")

    # 1) generate perturbations
    pert = Perturber(seed=args.seed)
    perts: List[Optional[str]] = []
    cats: List[Optional[str]] = []
    for tid, t in zip(df["text_id"], df["text"]):
        if not pert.is_eligible(t, min_numeric=2):
            perts.append(None); cats.append(None); continue
        out = pert.perturb(t, key=str(tid))
        if out is None:
            perts.append(None); cats.append(None)
        else:
            perts.append(out.text); cats.append(out.category)
    df["perturb_text"] = perts
    df["perturb_category"] = cats
    df["has_perturb"] = [p is not None for p in perts]
    n_pert = int(df["has_perturb"].sum())
    print(f"[extract] generated perturbations on {n_pert:,}/{len(df):,} rows ({n_pert/len(df):.1%})")

    # 2) encode with teacher
    model, tok = _load_teacher(args.teacher_model, args.dtype)
    t0 = time.time()
    print("[extract] encoding originals…")
    teacher_emb = _encode(df["text"].tolist(), model, tok, args.max_length, args.batch_size)
    print(f"[extract] originals done in {time.time()-t0:.1f}s -> shape {teacher_emb.shape}")

    pert_idx = df.index[df["has_perturb"]].tolist()
    pert_texts = df.loc[pert_idx, "perturb_text"].tolist()
    t1 = time.time()
    print(f"[extract] encoding {len(pert_texts):,} perturbations…")
    pert_emb = _encode(pert_texts, model, tok, args.max_length, args.batch_size)
    print(f"[extract] perturbations done in {time.time()-t1:.1f}s -> shape {pert_emb.shape}")

    full_pert_emb = np.zeros((len(df), teacher_emb.shape[1]), dtype="float32")
    full_pert_emb[pert_idx] = pert_emb
    pert_mask = np.zeros(len(df), dtype=bool); pert_mask[pert_idx] = True

    # 3) write parquet
    out = pd.DataFrame({
        "text_id": df["text_id"].values,
        "text": df["text"].values,
        "source": df["source"].values,
        "has_perturb": df["has_perturb"].values,
        "perturb_category": df["perturb_category"].values,
        "perturb_text": df["perturb_text"].values,
        "teacher_emb": [row.astype("float32") for row in teacher_emb],
        "teacher_perturb_emb": [row.astype("float32") if pert_mask[i] else None
                                for i, row in enumerate(full_pert_emb)],
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[extract] wrote {args.out}  ({len(out):,} rows, dim={teacher_emb.shape[1]})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--teacher_model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
