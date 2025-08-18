#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore properties of precomputed theta vectors:
 - Projection correlation with assessor scores
 - Δscore vs step size along theta
 - Variance profile across latent dimensions
 - Scatter of projection vs assessor score

Expects:
  /home/rothermm/THINGS/03_results/thetas/subjXX/
    - theta_emonet_decoded_top10_minus_bottom10.npy
    - theta_memnet_decoded_top10_minus_bottom10.npy
    - theta_summary_decoded_top10_minus_bottom10.json
"""

import argparse, pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, required=True, help="Subject number, e.g. 1")
    ap.add_argument("--scores_dir", type=Path, required=True,
        help="Directory containing emonet_scores_subXX.pkl, memnet_scores_subXX.pkl")
    ap.add_argument("--features_npz", type=Path, required=True,
        help="NPZ with test_latents")
    ap.add_argument("--out_dir", type=Path, default=None,
        help="Where to save plots (default: plots/theta_explore/subjXX)")
    args = ap.parse_args()

    sp = f"{args.sub:02d}"

    # auto theta dir
    theta_dir = Path(f"/home/rothermm/THINGS/03_results/thetas/subj{sp}")
    if not theta_dir.exists():
        raise FileNotFoundError(f"No theta dir: {theta_dir}")

    # output dir
    out_dir = args.out_dir or Path(f"/home/rothermm/THINGS/03_results/plots/theta_explore/subj{sp}")
    ensure_dir(out_dir)

    # --- load latents & scores
    ft = np.load(args.features_npz)
    if "test_latents" not in ft:
        raise KeyError("'test_latents' not found in features_npz")
    Z = ft["test_latents"]

    with open(args.scores_dir / f"emonet_scores_sub{sp}.pkl", "rb") as f:
        emo = pickle.load(f)
    with open(args.scores_dir / f"memnet_scores_sub{sp}.pkl", "rb") as f:
        mem = pickle.load(f)
    emo_scores = np.asarray(emo["decoded"], dtype=float)
    mem_scores = np.asarray(mem["decoded"], dtype=float)

    if not (len(emo_scores) == len(mem_scores) == Z.shape[0]):
        raise ValueError("Length mismatch between scores and test_latents")

    # --- load theta
    th_emo = np.load(theta_dir / "theta_emonet_decoded_top10_minus_bottom10.npy")
    th_mem = np.load(theta_dir / "theta_memnet_decoded_top10_minus_bottom10.npy")

    # --- projection correlation
    proj_emo = Z @ th_emo
    proj_mem = Z @ th_mem
    r_emo, p_emo = pearsonr(proj_emo, emo_scores)
    r_mem, p_mem = pearsonr(proj_mem, mem_scores)
    print(f"[EmoNet] corr(proj,score) = {r_emo:.3f} (p={p_emo:.1e})")
    print(f"[MemNet] corr(proj,score) = {r_mem:.3f} (p={p_mem:.1e})")

    # --- scatter plots
    for name, proj, scores, r, p in [
        ("emonet", proj_emo, emo_scores, r_emo, p_emo),
        ("memnet", proj_mem, mem_scores, r_mem, p_mem),
    ]:
        plt.figure(figsize=(5,5))
        sns.regplot(x=proj, y=scores, scatter_kws={"s":10, "alpha":0.6})
        plt.xlabel("Projection onto θ")
        plt.ylabel("Assessor score")
        plt.title(f"{name} (r={r:.2f}, p={p:.1e})")
        plt.tight_layout()
        plt.savefig(out_dir / f"theta_proj_scatter_{name}_sub{sp}.png", dpi=200)
        plt.close()

    # --- θ distribution
    for name, th in [("emonet", th_emo), ("memnet", th_mem)]:
        plt.figure()
        sns.histplot(np.abs(th), bins=50)
        plt.title(f"sub{sp} {name}: distribution of |θ| entries")
        plt.xlabel("|θ dimension|")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / f"theta_absdist_{name}_sub{sp}.png", dpi=200)
        plt.close()

    print("[DONE] Results written to", out_dir)

if __name__ == "__main__":
    main()
