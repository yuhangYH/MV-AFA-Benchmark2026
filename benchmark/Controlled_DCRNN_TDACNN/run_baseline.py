"""
run_baseline.py
===============
Train + evaluate DCRNN or TDA+CNN on CHB-MIT under the MV-AFA unified protocol,
producing exactly the metric set used for MV-AFA so the rows in Table "Controlled
baseline comparison" are protocol-compatible.

Usage
-----
    # patient-independent split (main controlled-comparison row)
    python run_baseline.py --model dcrnn   --cache chbmit_windows.npz --protocol pi
    python run_baseline.py --model tda_cnn --cache chbmit_windows.npz --protocol pi

    # LOSO (for the LOSO table); reports mean +/- std across folds
    python run_baseline.py --model dcrnn --cache chbmit_windows.npz --protocol loso

Expected cache (.npz), same windows MV-AFA consumes:
    X[N,18,512] float32, y[N], subject[N], rec_id[N], t_start[N],
    win_sec(=2.0), step_sec(=4.0)

The training loop is intentionally identical for both baselines (AdamW, lr 1e-4,
wd 1e-4, batch 16, dropout 0.2, 50 epochs, class-weighted CE, seed 42) to match
the MV-AFA training objective described in the paper.
"""
from __future__ import annotations
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore", message=".*output with one or more elements was resized.*")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import protocol as P


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# Model + input adapters
# ----------------------------------------------------------------------------
def build_model(name, n_channels):
    if name == "dcrnn":
        from dcrnn import DCRNN
        return DCRNN(n_channels=n_channels), "raw"
    if name == "tda_cnn":
        from tda_cnn import TDACNN
        return TDACNN(res=32, in_ch=2), "pimg"
    raise ValueError(name)


def make_inputs(cache: P.WindowCache, kind: str, res: int = 32):
    """Return a float tensor the model consumes."""
    if kind == "raw":
        return torch.from_numpy(cache.X)                   # [N, C, T]
    if kind == "pimg":
        from tda_cnn import windows_to_images_parallel
        imgs = windows_to_images_parallel(cache.X, res=res)
        return torch.from_numpy(imgs)                      # [N, 2, res, res]
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Train / predict
# ----------------------------------------------------------------------------
def train_model(model, Xtr, ytr, device, epochs=50, bs=16, lr=1e-4, wd=1e-4):
    model.to(device).train()
    # class-weighted cross-entropy (matches MV-AFA objective)
    cls_count = np.bincount(ytr.numpy(), minlength=2).astype(np.float32)
    w = torch.tensor(cls_count.sum() / (2 * cls_count + 1e-6),
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    # lazily build any lazy submodules (DCRNN) before optimizer sees them
    with torch.no_grad():
        _ = model(Xtr[:2].to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba(model, X, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device)
        p = torch.softmax(model(xb), dim=1)[:, 1]
        out.append(p.cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------
# Protocols
# ----------------------------------------------------------------------------
def run_pi(args, device):
    cache = P.WindowCache.load_npz(args.cache)
    tr, va, te = P.subject_disjoint_split(cache, seed=args.seed)
    tr_bal = P.balance_training(tr, seed=args.seed, neg_pos_ratio=args.neg_pos_ratio)

    model, kind = build_model(args.model, n_channels=cache.X.shape[1])
    Xtr = make_inputs(tr_bal, kind); ytr = torch.from_numpy(tr_bal.y)
    model = train_model(model, Xtr, ytr, device, epochs=args.epochs)

    Xva = make_inputs(va, kind)
    p_va = predict_proba(model, Xva, device)
    thr = P.select_threshold(va.y, p_va, strategy=args.threshold,
                             min_sensitivity=args.min_sens)

    Xte = make_inputs(te, kind)
    p_te = predict_proba(model, Xte, device)
    m = P.compute_metrics(te, p_te, thr)
    print(f"[{args.model} | PI] " + " ".join(f"{k}={v:.4f}"
          for k, v in m.items() if isinstance(v, float)))
    P.save_json({"model": args.model, "protocol": "pi", "metrics": m},
                args.out or f"{args.model}_pi.json")


def run_loso(args, device):
    cache = P.WindowCache.load_npz(args.cache)
    per_fold = []
    for tr, te, subj in P.loso_folds(cache):
        # train-side validation split (subject-disjoint within training pool)
        tr2, va, _ = P.subject_disjoint_split(tr, seed=args.seed,
                                               frac=(0.82, 0.18, 0.0))
        tr_bal = P.balance_training(tr2, seed=args.seed,
                                    neg_pos_ratio=args.neg_pos_ratio)
        model, kind = build_model(args.model, n_channels=cache.X.shape[1])
        Xtr = make_inputs(tr_bal, kind); ytr = torch.from_numpy(tr_bal.y)
        model = train_model(model, Xtr, ytr, device, epochs=args.epochs)

        p_va = predict_proba(model, make_inputs(va, kind), device)
        thr = P.select_threshold(va.y, p_va, strategy=args.threshold,
                                 min_sensitivity=args.min_sens)
        p_te = predict_proba(model, make_inputs(te, kind), device)
        m = P.compute_metrics(te, p_te, thr)
        m["subject"] = str(subj)
        per_fold.append(m)
        print(f"[{args.model} | LOSO fold {subj}] "
              f"AUROC={m['auroc']:.3f} F1={m['f1']:.3f} "
              f"DetRate={m['det_rate']:.3f} Delay={m['mean_delay']:.2f}")
    agg = P.aggregate_folds(per_fold)
    print(f"\n=== {args.model} LOSO (mean +/- std, n={len(per_fold)} folds) ===")
    for k in ("acc", "sens", "spec", "f1", "auroc", "auprc",
              "det_rate", "median_delay", "fa_per_hour"):
        print(f"  {k:13s} {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}")
    P.save_json({"model": args.model, "protocol": "loso",
                 "per_fold": per_fold, "aggregate": agg},
                args.out or f"{args.model}_loso.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["dcrnn", "tda_cnn"], required=True)
    ap.add_argument("--cache", required=True, help="windowed .npz (see header)")
    ap.add_argument("--protocol", choices=["pi", "loso"], default="pi")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neg_pos_ratio", type=float, default=1.0,
                    help="1.0 = balanced (main); 5.0 = LOSO opt runs")
    ap.add_argument("--threshold", default="sens_floor_spec",
                    choices=["f1", "youden", "sens_floor_spec"])
    ap.add_argument("--min_sens", type=float, default=0.70)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = _device()
    print(f"device={device}")
    (run_loso if args.protocol == "loso" else run_pi)(args, device)


if __name__ == "__main__":
    main()
