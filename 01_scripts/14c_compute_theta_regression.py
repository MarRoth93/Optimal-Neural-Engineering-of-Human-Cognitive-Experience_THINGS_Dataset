#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute regression-based theta from VDVAE latents -> assessor scores.

Reads:
  NPZ: {test_latents} (auto-detects among: test_latents, features, Z, latents)
  Scores: /.../assessor_scores/subjXX/{emonet,memnet}_scores_subXX.pkl
          -> choose --which {decoded,original}

Writes:
  /.../03_results/thetas_regression/subjXX/
    - theta_emonet_regression_<which>.npy
    - theta_memnet_regression_<which>.npy
    - theta_regression_summary_<which>.json
"""

import argparse, json, pickle
from pathlib import Path
import numpy as np

def load_latents(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    for k in ["test_latents", "features", "Z", "latents"]:
        if k in d: 
            Z = d[k]
            if isinstance(Z, np.ndarray):
                return Z
    raise KeyError(f"No expected latent key found in {npz_path}. Keys: {list(d.keys())}")

def zscore(X, axis=0, eps=1e-8):
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, keepdims=True)
    return (X - mu) / (sd + eps), mu.squeeze(), sd.squeeze()

def ridge_closed_form(X, y, alpha):
    # X: (N,D), y: (N,)
    # w = (X^T X + alpha I)^-1 X^T y
    XT = X.T
    A = XT @ X
    A.flat[:: A.shape[0]+1] += alpha  # add alpha to diagonal
    b = XT @ y
    w = np.linalg.solve(A, b)
    return w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, required=True)
    ap.add_argument("--features_npz", type=Path, required=True)
    ap.add_argument("--scores_dir", type=Path, required=True)
    ap.add_argument("--which", choices=["decoded","original"], default="decoded")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge L2 strength")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    sp = f"{args.sub:02d}"
    out_dir = args.out_dir or Path(f"/home/rothermm/THINGS/03_results/thetas_regression/subj{sp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- latents
    Z = load_latents(args.features_npz).astype(np.float32)  # (N,D)
    N, D = Z.shape
    X, mu, sd = zscore(Z, axis=0)                           # z-score per dim

    summary = {"subject": args.sub, "N": int(N), "D": int(D), "which": args.which, "alpha": args.alpha, "assessors": {}}

    for assessor in ["emonet", "memnet"]:
        pkl = args.scores_dir / f"{assessor}_scores_sub{sp}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"Missing: {pkl}")
        with open(pkl, "rb") as f:
            obj = pickle.load(f)
        y = np.asarray(obj[args.which], dtype=np.float32)
        if y.shape[0] != N:
            raise ValueError(f"Length mismatch for {assessor}: y={len(y)} vs Z={N}")

        # z-score target for stable ridge
        yz = (y - y.mean()) / (y.std() + 1e-8)

        # ridge (numpy closed-form; no sklearn needed)
        w = ridge_closed_form(X, yz, alpha=args.alpha)      # (D,)
        # normalize to a direction
        theta = (w / (np.linalg.norm(w) + 1e-12)).astype(np.float32)

        # save
        out_npy = out_dir / f"theta_{assessor}_regression_{args.which}.npy"
        np.save(out_npy, theta)

        # simple fit quality (in-sample r)
        yhat = X @ w
        r = float(np.corrcoef(yz, yhat)[0,1])
        summary["assessors"][assessor] = {
            "theta_path": str(out_npy),
            "theta_norm": float(np.linalg.norm(theta)),
            "corr_in_sample": r,
        }

    with open(out_dir / f"theta_regression_summary_{args.which}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] saved to", out_dir)

if __name__ == "__main__":
    main()
