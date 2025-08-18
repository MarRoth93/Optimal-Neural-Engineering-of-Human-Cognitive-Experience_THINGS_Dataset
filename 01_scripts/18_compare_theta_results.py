#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_compare_theta_results.py

Create publication-ready multi-panel comparisons per assessor (EmoNet/MemNet):
one figure per assessor with small multiples (one panel per alpha), plotting
Original vs. Alpha scores using shared axes, y=x reference, and regression fit.

Also creates a summary bar chart (mean±SD) for Original and all alphas.

Inputs (produced by compute_assessor_scores_shifted.py):
  /home/rothermm/THINGS/03_results/assessor_scores/subjXX/theta/
    - emonet_theta_scores_subXX.pkl
    - memnet_theta_scores_subXX.pkl

Outputs (this script):
  /home/rothermm/THINGS/03_results/plots/theta/
    - emonet_theta_compare_original_vs_allalphas_subXX.png
    - emonet_theta_summary_means_subXX.png
    - memnet_theta_compare_original_vs_allalphas_subXX.png
    - memnet_theta_summary_means_subXX.png

Usage:
  python scripts/analysis/18_compare_theta_results.py --sub 1
  python scripts/analysis/18_compare_theta_results.py --sub 1 --assessors emonet
  python scripts/analysis/18_compare_theta_results.py --sub 1 --limit 500
"""

import os
import sys
import math
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


# ---------- IO ----------

def load_pickle(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def default_score_dir_for_sub(sub: int) -> Path:
    sp = f"{sub:02d}"
    return Path(f"/home/rothermm/THINGS/03_results/assessor_scores/subj{sp}/theta")


def default_plot_dir_root() -> Path:
    return Path("/home/rothermm/THINGS/03_results/plots/theta")


# ---------- plotting helpers ----------

def setup_pub_style():
    sns.set_context("talk", rc={
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
    sns.set_style("whitegrid", rc={"grid.linestyle": "--", "axes.grid": True})


def compute_limits(x: np.ndarray, ys: List[np.ndarray], pad: float = 0.03) -> Tuple[float, float]:
    lo = np.min([x.min()] + [y.min() for y in ys])
    hi = np.max([x.max()] + [y.max() for y in ys])
    rng = hi - lo
    if rng <= 0:
        return float(lo - 1), float(hi + 1)
    return float(lo - pad * rng), float(hi + pad * rng)


def ncols_nrows(n: int, max_cols: int = 4) -> Tuple[int, int]:
    cols = min(max_cols, max(1, n))
    rows = math.ceil(n / cols)
    # Prefer squarer grids if possible
    while (cols - 1) >= 2 and (rows * (cols - 1)) >= n:
        cols -= 1
        rows = math.ceil(n / cols)
    return cols, rows


def add_panel(ax, x: np.ndarray, y: np.ndarray, title: str, lims: Tuple[float, float]):
    # scatter
    ax.scatter(x, y, s=8, alpha=0.5, linewidths=0, rasterized=True)
    # regression
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, deg=1)
        xx = np.linspace(lims[0], lims[1], 200)
        ax.plot(xx, coeffs[0]*xx + coeffs[1], linewidth=1.5)
    # y=x reference
    ax.plot(lims, lims, linestyle=":", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(title)
    ax.set_xlabel("Original")
    ax.set_ylabel("Shifted")


def save_tight(fig, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------- core figure builders ----------

def figure_small_multiples(data: dict,
                           assessor: str,
                           out_root: Path,
                           filename_stub: str,
                           limit: int = None) -> Path:
    """
    Build one multi-panel figure: Original vs each alpha.
    Saved as: {assessor}_theta_compare_original_vs_allalphas_{stub}.png
    """
    scores: Dict[str, List[float]] = data["scores"]
    original = np.asarray(scores["original"], dtype=np.float32)
    if limit is not None:
        original = original[:limit]

    # Collect alpha series; keep ALL alphas found
    alpha_keys = [k for k in scores.keys() if k.startswith("alpha_")]
    if not alpha_keys:
        raise ValueError("No alpha_* series found in the pickle.")

    def key_to_val(k: str) -> float:
        return float(k.split("alpha_")[1])

    alpha_keys = sorted(alpha_keys, key=key_to_val)

    # Prepare Y series and per-panel stats
    y_series = []
    panel_titles = []
    for k in alpha_keys:
        y = np.asarray(scores[k], dtype=np.float32)
        if limit is not None:
            y = y[:limit]
        r, p = pearsonr(original, y)
        y_series.append(y)
        a_val = key_to_val(k)
        # concise, readable panel header
        panel_titles.append(f"α={a_val:g}  (r={r:.2f}, p={p:.1e})")

    # Shared limits across all panels
    lims = compute_limits(original, y_series, pad=0.04)

    # Layout
    setup_pub_style()
    cols, rows = ncols_nrows(len(alpha_keys), max_cols=4)
    fig_w = 5.0 * cols
    fig_h = 4.6 * rows + 1.6  # extra space for suptitle
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    # Plot panels
    for idx, (y, title) in enumerate(zip(y_series, panel_titles)):
        r = idx // cols
        c = idx % cols
        add_panel(axes[r][c], original, y, title, lims)

    # Hide unused axes
    for idx in range(len(alpha_keys), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    subj = data.get("subject", None)
    n = data.get("n", len(original))
    fig.suptitle(f"{assessor.capitalize()} — Original vs Shifted (subj{subj:02d}, N={n})", y=0.995)

    out_path = out_root / f"{assessor}_theta_compare_original_vs_allalphas_{filename_stub}.png"
    save_tight(fig, out_path, dpi=300)
    return out_path


def figure_summary_bars(data: dict,
                        assessor: str,
                        out_root: Path,
                        filename_stub: str) -> Path:
    """
    Bar chart of means ± SD for Original and all alphas.
    Saved as: {assessor}_theta_summary_means_{stub}.png
    """
    means = data["means"]
    stds = data["stds"]

    # Ordered labels: Original, then alphas sorted by numeric value
    alpha_items = []
    for k in means.keys():
        if k.startswith("alpha_"):
            val = float(k.split("alpha_")[1])
            alpha_items.append((val, k))
    alpha_items.sort(key=lambda x: x[0])

    labels = ["original"] + [k for _, k in alpha_items]
    mu = [means[l] for l in labels]
    sd = [stds[l] for l in labels]

    setup_pub_style()
    fig, ax = plt.subplots(figsize=(max(6, 0.8*len(labels)+2), 4.0))
    x = np.arange(len(labels))
    ax.bar(x, mu, yerr=sd, capsize=3)
    ax.set_xticks(x)
    xt = []
    for l in labels:
        xt.append("Original" if l == "original" else f"α={l.split('alpha_')[1]}")
    ax.set_xticklabels(xt, rotation=0)
    ax.set_ylabel("Score (mean ± SD)")
    subj = data.get("subject", None)
    ax.set_title(f"{assessor.capitalize()} — Summary of Score Distributions (subj{subj:02d})")

    out_path = out_root / f"{assessor}_theta_summary_means_{filename_stub}.png"
    save_tight(fig, out_path, dpi=300)
    return out_path


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, required=True, help="Subject number, e.g., 1")
    ap.add_argument("--scores_dir", type=Path, default=None,
                    help="Directory with *_theta_scores_subXX.pkl (defaults to THINGS layout)")
    ap.add_argument("--assessors", choices=["both", "emonet", "memnet"], default="both",
                    help="Which assessors to render (default: both).")
    ap.add_argument("--limit", type=int, default=None, help="Limit points for quick drafts.")
    ap.add_argument("--out_dir", type=Path, default=default_plot_dir_root(),
                    help="Output directory for all plots (default: /home/rothermm/THINGS/03_results/plots/theta)")
    args = ap.parse_args()

    scores_dir = args.scores_dir or default_score_dir_for_sub(args.sub)
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    todo = []
    if args.assessors in ("both", "emonet"):
        todo.append("emonet")
    if args.assessors in ("both", "memnet"):
        todo.append("memnet")

    for assessor in todo:
        pkl = scores_dir / f"{assessor}_theta_scores_sub{args.sub:02d}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(f"Missing: {pkl}")
        data = load_pickle(pkl)

        # Use a compact, intuitive filename stub that includes subject
        stub = f"sub{args.sub:02d}"

        print(f"[{assessor}] Building small-multiples figure…")
        fig_path = figure_small_multiples(
            data=data,
            assessor=assessor,
            out_root=out_root,
            filename_stub=stub,
            limit=args.limit,
        )
        print(f"[{assessor}] Saved: {fig_path}")

        print(f"[{assessor}] Building summary means figure…")
        summ_path = figure_summary_bars(
            data=data,
            assessor=assessor,
            out_root=out_root,
            filename_stub=stub
        )
        print(f"[{assessor}] Saved: {summ_path}")

    print("[DONE] All plots written to:", out_root)


if __name__ == "__main__":
    main()
