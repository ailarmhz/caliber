#!/usr/bin/env python3
"""Training entry point: load a YAML config (keys map to CaliberConfig) and run train()."""
import argparse
import os
import sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from caliber.train import CaliberConfig, train


def _coerce(v):
    if isinstance(v, str):
        if v.lower() in ("true", "false"): return v.lower() == "true"
        try: return int(v)
        except ValueError: pass
        try: return float(v)
        except ValueError: pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="key=value pairs to override the YAML")
    args = ap.parse_args()
    with open(args.config) as f:
        d = yaml.safe_load(f)
    for kv in args.override:
        k, v = kv.split("=", 1); d[k] = _coerce(v)
    cfg = CaliberConfig(**d)
    print("[caliber] config:")
    for k, v in cfg.__dict__.items():
        print(f"  {k} = {v}")
    train(cfg)


if __name__ == "__main__":
    main()
