#!/usr/bin/env python3
"""Aggregate per-evaluator CSVs under a runs directory into a single results table.

Collects retrieval metrics (eval_asym.py, eval_mteb.py), NumGap metrics
(eval_numgap.py), and per-epoch training history, and writes results_table.csv
plus a per-category NumGap breakdown and a validation-loss summary.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def collect(args):
    rows = defaultdict(dict)  # key=(model_name) -> dict of metrics

    # FinDER + asymmetric eval
    for csvf in glob.glob(os.path.join(args.runs_dir, "**", "metrics_finder*.csv"), recursive=True):
        with open(csvf) as f:
            for r in csv.DictReader(f):
                key = (r["model"], r["mode"], "FinDER")
                rows[key].update(r)
                rows[key]["benchmark"] = "FinDER"

    # FinanceBench
    for csvf in glob.glob(os.path.join(args.runs_dir, "**", "metrics_fb*.csv"), recursive=True):
        with open(csvf) as f:
            for r in csv.DictReader(f):
                key = (r["model"], r["mode"], "FinanceBench")
                rows[key].update(r); rows[key]["benchmark"] = "FinanceBench"

    # NumGap
    for csvf in glob.glob(os.path.join(args.runs_dir, "**", "numgap_metrics*.csv"), recursive=True):
        with open(csvf) as f:
            for r in csv.DictReader(f):
                if r["category"] != "overall": continue
                key = (r["model"], "student/student", "NumGap")
                rows[key].update({"model": r["model"], "mode": "student/student",
                                  "benchmark": "NumGap", "main_score": r["numgap_d"],
                                  "numgap_d": r["numgap_d"], "numgap_m": r["numgap_m"]})

    # MTEB
    for csvf in glob.glob(os.path.join(args.runs_dir, "**", "mteb_summary*.csv"), recursive=True):
        with open(csvf) as f:
            for r in csv.DictReader(f):
                key = (r["model"], "student/student", r["task"])
                rows[key].update(r); rows[key]["benchmark"] = r["task"]; rows[key]["mode"] = "student/student"

    # Per-category NumGap (write a separate breakdown file)
    cat_rows = []
    for csvf in glob.glob(os.path.join(args.runs_dir, "**", "numgap_metrics*.csv"), recursive=True):
        with open(csvf) as f:
            for r in csv.DictReader(f):
                cat_rows.append(r)

    # Robustness margin from history.json
    margin_rows = []
    for hist in glob.glob(os.path.join(args.runs_dir, "**", "history.json"), recursive=True):
        with open(hist) as f:
            history = json.load(f)
        run = os.path.basename(os.path.dirname(hist))
        for h in history:
            margin_rows.append({"run": run, "epoch": h["epoch"],
                                "val_loss": h.get("val_loss"),
                                "val_L_align_orig": h.get("val_L_align_orig")})

    # Write master table
    out_rows = []
    for (model, mode, bench), v in rows.items():
        out_rows.append({"model": model, "mode": mode, "benchmark": bench,
                         "main_score": v.get("main_score") or v.get("ndcg_at_10"),
                         "ndcg_at_10": v.get("ndcg_at_10"),
                         "recall_at_10": v.get("recall_at_10"),
                         "recall_at_100": v.get("recall_at_100"),
                         "numgap_d": v.get("numgap_d"),
                         "numgap_m": v.get("numgap_m")})

    if out_rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader()
            for r in out_rows: w.writerow(r)
        print(f"[collect] wrote {len(out_rows)} rows to {args.out}")
    else:
        print("[collect] WARNING: no rows; check that metrics CSVs exist under", args.runs_dir)

    if cat_rows:
        cat_csv = args.out.replace(".csv", "_numgap_categories.csv")
        with open(cat_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cat_rows[0].keys())); w.writeheader()
            for r in cat_rows: w.writerow(r)
        print(f"[collect] per-category NumGap -> {cat_csv}")

    if margin_rows:
        m_csv = args.out.replace(".csv", "_robustness_margin.csv")
        with open(m_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(margin_rows[0].keys())); w.writeheader()
            for r in margin_rows: w.writerow(r)
        print(f"[collect] robustness-margin checkpoints -> {m_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--out", default="results_table.csv")
    args = p.parse_args()
    collect(args)
