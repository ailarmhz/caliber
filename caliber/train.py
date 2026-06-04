"""Caliber training: L2 alignment to a frozen teacher plus a numeric-faithfulness hinge.

Per-sample loss:
    L_align(x)      = || s(x) - t(x) ||_2
    L_align(pi(x))  = || s(pi(x)) - t(pi(x)) ||_2        (only when a perturbation exists)
    L_num           = max(0, Delta_t + m - Delta_s)      (only when a perturbation exists)
        Delta_t = 1 - cos(t(x), t(pi(x)))   (teacher discrimination, precomputed)
        Delta_s = 1 - cos(s(x), s(pi(x)))   (student discrimination)

    total = L_align(x) + 1[has_perturb] * (L_align(pi(x)) + lambda_num * L_num)

Setting lambda_num = 0 recovers the LEAF alignment-only baseline.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class CaliberConfig:
    student_init_model_path: str
    teacher_parquet: str
    output_dir: str

    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_frac: float = 0.05

    lambda_num: float = 0.5
    margin: float = 0.05
    perturb_weight: float = 1.0

    max_length: int = 512
    padding_side: str = "right"

    use_amp: bool = True
    model_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "sdpa"
    gradient_checkpointing: bool = False

    val_frac: float = 0.02
    save_every_epoch: bool = True
    log_every: int = 25
    seed: int = 42


class CaliberDataset(Dataset):
    """Reads the parquet written by extract_teacher_perturb.py.

    When ``truncate_dim`` is smaller than the stored teacher width, teacher
    embeddings are sliced to that prefix and re-normalized. Qwen3-Embedding is
    trained with Matryoshka representation learning, so the renormalized first
    k coordinates are the canonical k-dimensional embedding.
    """

    def __init__(self, df: pd.DataFrame, truncate_dim: Optional[int] = None):
        self.df = df.reset_index(drop=True)
        self.truncate_dim = truncate_dim

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _trunc_norm(v: np.ndarray, k: Optional[int]) -> np.ndarray:
        if k is None or k >= v.shape[-1]:
            return v
        v = v[:k]
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        return v

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        t = self._trunc_norm(np.asarray(r["teacher_emb"], dtype=np.float32), self.truncate_dim)
        if r["has_perturb"]:
            tp = self._trunc_norm(np.asarray(r["teacher_perturb_emb"], dtype=np.float32), self.truncate_dim)
        else:
            tp = np.zeros_like(t)
        return {
            "text": r["text"],
            "teacher_emb": t,
            "has_perturb": bool(r["has_perturb"]),
            "perturb_text": r["perturb_text"] if r["has_perturb"] else "",
            "perturb_emb": tp,
        }


def caliber_collate(batch):
    return {
        "texts": [b["text"] for b in batch],
        "teacher_emb": torch.from_numpy(np.stack([b["teacher_emb"] for b in batch])),
        "has_perturb": torch.tensor([b["has_perturb"] for b in batch], dtype=torch.bool),
        "perturb_texts": [b["perturb_text"] for b in batch],
        "perturb_emb": torch.from_numpy(np.stack([b["perturb_emb"] for b in batch])),
    }


def _dtype_from(s: str):
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[s]


def load_student(cfg: CaliberConfig, device: torch.device):
    tok = AutoTokenizer.from_pretrained(cfg.student_init_model_path, trust_remote_code=False)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = cfg.padding_side
    kwargs = {"torch_dtype": _dtype_from(cfg.model_dtype)}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    model = AutoModel.from_pretrained(cfg.student_init_model_path, trust_remote_code=False, **kwargs).to(device)
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tok


def encode_student(model, tok, texts: List[str], cfg: CaliberConfig, device) -> torch.Tensor:
    inp = tok(texts, padding=True, truncation=True, max_length=cfg.max_length,
              return_tensors="pt").to(device)
    h = model(**inp).last_hidden_state
    mask = inp["attention_mask"].unsqueeze(-1).to(h.dtype)
    emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return F.normalize(emb, p=2, dim=-1)


def caliber_loss(s_emb, t_emb, s_pert_emb, t_pert_emb, has_pert, lam, margin, w_pert):
    """All inputs are L2-normalized. Returns (total, component dict)."""
    L_align_orig = torch.norm(s_emb - t_emb, p=2, dim=-1).mean()

    if has_pert.any():
        idx = has_pert
        s_p, t_p = s_pert_emb[idx], t_pert_emb[idx]
        s_o, t_o = s_emb[idx], t_emb[idx]
        L_align_pert = torch.norm(s_p - t_p, p=2, dim=-1).mean()

        delta_t = 1.0 - (t_o * t_p).sum(dim=-1)
        delta_s = 1.0 - (s_o * s_p).sum(dim=-1)
        L_num = torch.relu(delta_t + margin - delta_s).mean()
    else:
        L_align_pert = torch.tensor(0.0, device=s_emb.device)
        L_num = torch.tensor(0.0, device=s_emb.device)

    total = L_align_orig + w_pert * L_align_pert + lam * L_num
    return total, {
        "L_align_orig": float(L_align_orig.detach().cpu()),
        "L_align_pert": float(L_align_pert.detach().cpu()),
        "L_num": float(L_num.detach().cpu()),
    }


def _make_split(df: pd.DataFrame, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(df)
    n_val = max(64, int(round(n * val_frac)))
    idx = np.arange(n)
    rng.shuffle(idx)
    return df.iloc[idx[n_val:]].copy(), df.iloc[idx[:n_val]].copy()


@torch.no_grad()
def evaluate_val(model, tok, val_loader, cfg, device, lam, margin, w_pert):
    model.eval()
    sum_loss = sum_lo = sum_lp = sum_ln = 0.0
    n = 0
    for batch in val_loader:
        s_o = encode_student(model, tok, batch["texts"], cfg, device)
        t_o = batch["teacher_emb"].to(device, non_blocking=True)

        if batch["has_perturb"].any():
            mask = batch["has_perturb"].to(device)
            pt = [t for t, h in zip(batch["perturb_texts"], batch["has_perturb"]) if h]
            s_p_compact = encode_student(model, tok, pt, cfg, device)
            s_p = torch.zeros_like(s_o)
            s_p[mask] = s_p_compact
            t_p = batch["perturb_emb"].to(device)
        else:
            mask = torch.zeros(len(batch["texts"]), dtype=torch.bool, device=device)
            s_p = torch.zeros_like(s_o)
            t_p = batch["perturb_emb"].to(device)

        loss, parts = caliber_loss(s_o, t_o, s_p, t_p, mask, lam, margin, w_pert)
        sum_loss += float(loss)
        sum_lo += parts["L_align_orig"]
        sum_lp += parts["L_align_pert"]
        sum_ln += parts["L_num"]
        n += 1
    model.train()
    if n == 0:
        return {}
    return {"val_loss": sum_loss / n, "val_L_align_orig": sum_lo / n,
            "val_L_align_pert": sum_lp / n, "val_L_num": sum_ln / n}


def train(cfg: CaliberConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} dtype={cfg.model_dtype} amp={cfg.use_amp}")

    df = pd.read_parquet(cfg.teacher_parquet)
    logger.info(f"loaded {len(df):,} rows; perturbations on {int(df['has_perturb'].sum()):,}")
    train_df, val_df = _make_split(df, cfg.val_frac, cfg.seed)

    model, tok = load_student(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable params: {n_params:,}")

    # Teacher is wider than the student (4096 vs 1024 for Qwen3-Embedding); slice
    # the teacher to the student width via the Matryoshka prefix and renormalize.
    student_dim = int(model.config.hidden_size)
    teacher_dim = int(np.asarray(df.iloc[0]["teacher_emb"]).shape[-1])
    truncate_dim = student_dim if teacher_dim > student_dim else None
    logger.info(f"student_dim={student_dim} teacher_dim={teacher_dim} truncate_dim={truncate_dim}")

    train_loader = DataLoader(CaliberDataset(train_df, truncate_dim=truncate_dim),
                              batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=2, collate_fn=caliber_collate, pin_memory=True)
    val_loader = DataLoader(CaliberDataset(val_df, truncate_dim=truncate_dim),
                            batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                            num_workers=1, collate_fn=caliber_collate, pin_memory=True)
    logger.info(f"train batches: {len(train_loader)}; val batches: {len(val_loader)}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.num_epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_frac * total_steps)

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 1.0 - 0.9 * prog)

    autocast = torch.amp.autocast
    history: List[dict] = []
    step = 0
    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        sum_total = sum_lo = sum_lp = sum_ln = 0.0
        nb = 0
        for bi, batch in enumerate(train_loader, start=1):
            step += 1
            for g in opt.param_groups:
                g["lr"] = cfg.learning_rate * lr_at(step)

            with autocast("cuda", enabled=cfg.use_amp, dtype=torch.bfloat16):
                s_o = encode_student(model, tok, batch["texts"], cfg, device)
                t_o = batch["teacher_emb"].to(device, non_blocking=True)
                if batch["has_perturb"].any():
                    mask = batch["has_perturb"].to(device)
                    pt = [t for t, h in zip(batch["perturb_texts"], batch["has_perturb"]) if h]
                    s_p_compact = encode_student(model, tok, pt, cfg, device)
                    s_p = torch.zeros_like(s_o)
                    s_p[mask] = s_p_compact
                    t_p = batch["perturb_emb"].to(device)
                else:
                    mask = torch.zeros(len(batch["texts"]), dtype=torch.bool, device=device)
                    s_p = torch.zeros_like(s_o)
                    t_p = batch["perturb_emb"].to(device)

                loss, parts = caliber_loss(s_o, t_o, s_p, t_p, mask,
                                           cfg.lambda_num, cfg.margin, cfg.perturb_weight)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            opt.step()

            sum_total += float(loss)
            sum_lo += parts["L_align_orig"]
            sum_lp += parts["L_align_pert"]
            sum_ln += parts["L_num"]
            nb += 1
            if bi % cfg.log_every == 0 or bi == len(train_loader):
                logger.info(f"e{epoch} step {step} bi {bi}/{len(train_loader)} "
                            f"loss={float(loss):.4f} lo={parts['L_align_orig']:.4f} "
                            f"lp={parts['L_align_pert']:.4f} ln={parts['L_num']:.4f} "
                            f"lr={opt.param_groups[0]['lr']:.2e}")

        val = evaluate_val(model, tok, val_loader, cfg, device,
                           cfg.lambda_num, cfg.margin, cfg.perturb_weight)
        rec = {
            "epoch": epoch,
            "train_loss": sum_total / max(1, nb),
            "train_L_align_orig": sum_lo / max(1, nb),
            "train_L_align_pert": sum_lp / max(1, nb),
            "train_L_num": sum_ln / max(1, nb),
            "epoch_time_s": time.time() - t0,
            **val,
        }
        history.append(rec)
        logger.info(f"epoch {epoch} done: {rec}")

        if cfg.save_every_epoch:
            ck = os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch}")
            model.save_pretrained(ck)
            tok.save_pretrained(ck)
            with open(os.path.join(ck, "metrics.json"), "w") as f:
                json.dump(rec, f, indent=2)
            logger.info(f"saved {ck}")

    final = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"done. final -> {final}")
