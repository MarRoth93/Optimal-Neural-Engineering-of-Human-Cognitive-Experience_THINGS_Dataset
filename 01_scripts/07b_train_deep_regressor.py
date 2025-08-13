#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a deep regressor (MLP with residual blocks) to map fMRI -> VDVAE latents.
Writes predicted test latents de-standardized to the original latent space.

Usage (defaults for subj01):
python 07b_train_deep_regressor.py --sub 1
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ----------------- utils -----------------
def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return mu, sd

def zscore_apply(X, mu, sd):
    return (X - mu) / sd

def inv_zscore(Y, mu, sd):
    return Y * sd + mu

def set_seed(s=0):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ----------------- model -----------------
class ResidualBlock(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d*4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d*4, d),
        )
    def forward(self, x):
        return x + self.net(x)

class Regressor(nn.Module):
    def __init__(self, in_dim, out_dim, width=2048, depth=6, drop=0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.GELU()]
        for _ in range(depth):
            layers.append(ResidualBlock(width, drop))
        layers += [nn.LayerNorm(width), nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ----------------- training -----------------
def cosine_mse_loss(pred, target, w_cos=1.0, w_mse=1.0):
    cos = nn.functional.cosine_similarity(pred, target, dim=1)
    cos_loss = 1 - cos.mean()
    mse_loss = nn.functional.mse_loss(pred, target)
    return w_cos*cos_loss + w_mse*mse_loss, {"cos": cos_loss.item(), "mse": mse_loss.item()}

def run(args):
    set_seed(args.seed)
    sub_pad = f"{args.sub:02d}"

    # ---------- paths ----------
    features_npz = args.features_npz or f"/home/rothermm/THINGS/02_data/extracted_features/subj{sub_pad}/things_vdvae_features_31l.npz"
    train_fmri_p = args.train_fmri or f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sub_pad}/X_train.npy"
    test_fmri_p  = args.test_fmri  or f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sub_pad}/X_test_avg.npy"
    out_pred_dir = Path(args.out_pred_dir or f"/home/rothermm/THINGS/02_data/predicted_features/subj{sub_pad}")
    out_ckpt_dir = Path(args.out_ckpt_dir or f"/home/rothermm/THINGS/02_data/learned_models/subj{sub_pad}")
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load ----------
    ft = np.load(features_npz)
    Y_tr_raw = ft["train_latents"]   # (Ntr, D)
    Y_te_true = ft["test_latents"]   # (Nte, D) optional for reporting

    X_tr = np.load(train_fmri_p)     # (Ntr, V)
    X_te = np.load(test_fmri_p)      # (Nte, V)

    # ---------- preprocess fMRI (match your pipeline) ----------
    X_tr = X_tr / 300.0
    X_te = X_te / 300.0
    x_mu, x_sd = zscore_fit(X_tr)
    X_tr = zscore_apply(X_tr, x_mu, x_sd)
    X_te = zscore_apply(X_te, x_mu, x_sd)

    # Optional: voxel selection by variance (keeps highest-var voxels)
    if args.keep_voxels and 0 < args.keep_voxels < X_tr.shape[1]:
        var = np.var(X_tr, axis=0)
        keep_idx = np.argsort(var)[-args.keep_voxels:]
        keep_idx.sort()
        X_tr = X_tr[:, keep_idx]
        X_te = X_te[:, keep_idx]
    else:
        keep_idx = None

    # ---------- standardize latents for training ----------
    y_mu, y_sd = zscore_fit(Y_tr_raw)
    Y_tr = zscore_apply(Y_tr_raw, y_mu, y_sd)

    # ---------- tensors/dataloaders ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(X_tr).float()
    Ytr = torch.from_numpy(Y_tr).float()
    Xte = torch.from_numpy(X_te).float()
    ds_tr = TensorDataset(Xtr, Ytr)

    n_train = len(ds_tr)
    n_val = max(int(n_train * args.val_frac), 1)
    n_fit = n_train - n_val
    tr_ds, va_ds = torch.utils.data.random_split(ds_tr, [n_fit, n_val], generator=torch.Generator().manual_seed(args.seed))

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=1, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # ---------- model/opt ----------
    model = Regressor(in_dim=X_tr.shape[1], out_dim=Y_tr.shape[1],
                      width=args.width, depth=args.depth, drop=args.drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    steps_per_epoch = max(len(tr_loader), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                epochs=args.epochs, steps_per_epoch=steps_per_epoch)

    best_val = float("inf")
    best_path = out_ckpt_dir / f"dnn_regressor_best_sub{sub_pad}.pt"
    mse_warmup_epochs = 5
    patience = 0

    # ---------- train ----------
    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss_acc = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            if args.input_noise > 0:
                xb = xb + torch.randn_like(xb) * args.input_noise
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(xb)
                mse_weight = args.w_mse if epoch > mse_warmup_epochs else 0.0
                loss, _ = cosine_mse_loss(pred, yb, w_cos=args.w_cos, w_mse=mse_weight)

            scaler.scale(loss).backward()
            # clip with AMP: unscale first
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tr_loss_acc += loss.item() * xb.size(0)

        # validation (use full objective)
        model.eval()
        va_loss_acc = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss, _ = cosine_mse_loss(pred, yb, w_cos=args.w_cos, w_mse=args.w_mse)
                va_loss_acc += loss.item() * xb.size(0)

        tr_loss = tr_loss_acc / max(1, len(tr_ds))
        va_loss = va_loss_acc / max(1, len(va_ds))
        print(f"[{epoch:03d}] train={tr_loss:.4f}  val={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "config": vars(args),
                "x_mu": x_mu, "x_sd": x_sd,
                "y_mu": y_mu, "y_sd": y_sd,
                "keep_idx": keep_idx,
            }, best_path)
            patience = 0
        else:
            patience += 1
        if args.early_stop > 0 and patience >= args.early_stop:
            print(f"[EARLY STOP] no val improvement for {patience} epochs.")
            break

    # ---------- load best & predict ----------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        Yte_hat_std = []
        for i in range(0, Xte.size(0), args.batch_size):
            xb = Xte[i:i+args.batch_size].to(device)
            Yte_hat_std.append(model(xb).cpu().numpy())
        Yte_hat_std = np.vstack(Yte_hat_std)

    # de-standardize to original latent space (so decoder can use it)
    Yte_hat = inv_zscore(Yte_hat_std, y_mu, y_sd)

    # optional R^2 against true test latents (if available)
    test_r2_mean = None
    if Y_te_true is not None and Y_te_true.shape == Yte_hat.shape:
        ss_res = ((Y_te_true - Yte_hat)**2).sum(axis=0)
        ss_tot = ((Y_te_true - Y_te_true.mean(axis=0))**2).sum(axis=0)
        r2_per_dim = 1 - ss_res/np.maximum(ss_tot, 1e-12)
        test_r2_mean = float(r2_per_dim.mean())
        r2_median = float(np.median(r2_per_dim))
        q25, q75 = float(np.percentile(r2_per_dim,25)), float(np.percentile(r2_per_dim,75))
        print(f"[TEST] mean R^2: {test_r2_mean:.4f} | median: {r2_median:.4f} | 25/75%: {q25:.4f}/{q75:.4f}")
        np.save(out_ckpt_dir / f"r2_per_latent_sub{sub_pad}.npy", r2_per_dim)

    # ---------- save predictions ----------
    tag = f"DNN_w{args.width}_d{args.depth}"
    out_pred_path = out_pred_dir / f"things_vdvae_pred_sub{sub_pad}_31l_{tag}.npy"
    np.save(out_pred_path, Yte_hat)
    print(f"[DONE] saved predictions -> {out_pred_path}")

    # Save training summary
    with open(out_ckpt_dir / f"dnn_regressor_summary_sub{sub_pad}.json", "w") as f:
        json.dump({
            "best_val_loss": best_val,
            "test_mean_r2": test_r2_mean,
            "pred_path": str(out_pred_path),
            "keep_idx": None if keep_idx is None else keep_idx.tolist(),
        }, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--features_npz", type=str, default=None)
    ap.add_argument("--train_fmri", type=str, default=None)
    ap.add_argument("--test_fmri",  type=str, default=None)
    ap.add_argument("--out_pred_dir", type=str, default=None)
    ap.add_argument("--out_ckpt_dir", type=str, default=None)
    # model/training
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--early_stop", type=int, default=10)
    ap.add_argument("--w_cos", type=float, default=1.0)
    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--keep_voxels", type=int, default=0, help="0=keep all; else top-K by variance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input_noise", type=float, default=0.01)  # try 0.01–0.05
    args = ap.parse_args()
    run(args)
