#!/usr/bin/env python3
"""Run MTEB retrieval tasks (e.g. FinanceBenchRetrieval) on a checkpoint.

Wraps a Hugging Face model behind the encode() interface MTEB expects, with
optional query/passage prefixes. MTEB writes per-task JSON to out_dir; this
script also extracts the headline metrics into mteb_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class HFEmbeddingModel:
    """Adapter that exposes mteb's encode() interface."""
    def __init__(self, model_path: str, max_length: int = 512, batch_size: int = 32,
                 prompt_query: str = "", prompt_passage: str = ""):
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=False)
        self.model.eval()
        if torch.cuda.is_available(): self.model = self.model.to("cuda")
        self.device = next(self.model.parameters()).device
        self.max_length = max_length; self.batch_size = batch_size
        self.prompt_query = prompt_query; self.prompt_passage = prompt_passage

    @torch.no_grad()
    def _encode(self, texts: List[str], prompt: str = "") -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            b = [prompt + t if prompt else t for t in texts[i:i + self.batch_size]]
            inp = self.tok(b, padding=True, truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            h = self.model(**inp).last_hidden_state
            mask = inp["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            emb = F.normalize(emb, p=2, dim=-1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1024), dtype="float32")

    def encode(self, sentences, **kwargs): return self._encode(list(sentences), self.prompt_query)
    def encode_queries(self, queries, **kwargs): return self._encode(list(queries), self.prompt_query)
    def encode_corpus(self, corpus, **kwargs):
        if isinstance(corpus, list) and corpus and isinstance(corpus[0], dict):
            texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in corpus]
        else:
            texts = list(corpus)
        return self._encode(texts, self.prompt_passage)


def main(args):
    import mteb

    model = HFEmbeddingModel(args.model, max_length=args.max_length,
                             batch_size=args.batch_size,
                             prompt_query=args.prompt_query,
                             prompt_passage=args.prompt_passage)

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = [mteb.get_task(name) for name in args.tasks]
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=args.out_dir, eval_splits=["test"])

    # Aggregate headline numbers
    summary_rows = []
    for task_name in args.tasks:
        result_files = [f for f in os.listdir(args.out_dir) if f.endswith(".json") and task_name in f]
        for rf in result_files:
            with open(os.path.join(args.out_dir, rf)) as f:
                data = json.load(f)
            scores = data.get("scores", {}) or data.get("test", {})
            row = {"model": os.path.basename(os.path.normpath(args.model)), "task": task_name}
            # Pull common keys defensively
            def deep_get(d, *paths):
                for p in paths:
                    try:
                        cur = d
                        for k in p.split("."):
                            cur = cur[k]
                        return cur
                    except Exception:
                        continue
                return None
            row["main_score"] = deep_get(data, "scores.test.main_score",
                                         "test.main_score", "main_score")
            row["ndcg_at_10"] = deep_get(data, "scores.test.ndcg_at_10",
                                         "test.ndcg_at_10", "ndcg_at_10")
            row["recall_at_10"] = deep_get(data, "scores.test.recall_at_10",
                                            "test.recall_at_10", "recall_at_10")
            row["recall_at_100"] = deep_get(data, "scores.test.recall_at_100",
                                             "test.recall_at_100", "recall_at_100")
            summary_rows.append(row)

    summary_csv = os.path.join(args.out_dir, "mteb_summary.csv")
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows: w.writerow(r)
        print(f"[mteb] summary -> {summary_csv}")
    else:
        print("[mteb] no headline rows extracted; check the per-task JSONs in", args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", nargs="+", default=["FinanceBenchRetrieval"])
    p.add_argument("--out_dir", default="mteb_results")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--prompt_query", default="")
    p.add_argument("--prompt_passage", default="")
    args = p.parse_args()
    main(args)
