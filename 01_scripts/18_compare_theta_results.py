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
from scipy.stats import linregress



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


def figure_alpha_effects_vs_alpha0(
    data: dict,
    assessor: str,
    out_root: Path,
    filename_stub: str,
    limit: int = None,
    use_sem: bool = True,
) -> Path:
    """
    Plot how scores change vs alpha=0 and how many images exceed the alpha=0 mean.

    Figure layout (2 rows):
      (1) Δmean(α) = mean(score@α) − mean(score@α=0), with error bars and
          separate linear fits (α<0 and α>0).
      (2) Proportion above μ0 = mean(score@α=0):  mean( 1[ score@α > μ0 ] )
          also with separate trend fits.

    Saves:
      {assessor}_theta_alpha_effects_vs_alpha0_{stub}.png
    """
    scores: Dict[str, List[float]] = data["scores"]

    # --- collect alpha series ---
    def is_alpha_key(k: str) -> bool:
        return k.startswith("alpha_") or k == "alpha0"  # be tolerant

    # map keys -> numeric alpha
    alpha_pairs = []
    for k in scores.keys():
        if is_alpha_key(k):
            if k == "alpha0":
                aval = 0.0
            else:
                try:
                    aval = float(k.split("alpha_")[1])
                except Exception:
                    continue
            alpha_pairs.append((aval, k))
    if not alpha_pairs:
        raise ValueError("No alpha series found (alpha_* or alpha0).")

    alpha_pairs.sort(key=lambda x: x[0])
    alpha_vals = []
    series = []
    for aval, k in alpha_pairs:
        arr = np.asarray(scores[k], dtype=np.float32)
        if limit is not None:
            arr = arr[:limit]
        alpha_vals.append(aval)
        series.append(arr)
    alpha_vals = np.asarray(alpha_vals, dtype=np.float32)  # shape (A,)

    # --- choose reference: α=0 preferred; fallback to 'original' ---
    ref_idx = None
    for i, a in enumerate(alpha_vals):
        if np.isclose(a, 0.0):
            ref_idx = i
            break
    used_original = False
    if ref_idx is None:
        if "original" not in scores:
            raise ValueError("Neither alpha=0 nor 'original' found to use as reference.")
        ref = np.asarray(scores["original"], dtype=np.float32)
        if limit is not None:
            ref = ref[:limit]
        mu0 = float(np.mean(ref))
        used_original = True
    else:
        mu0 = float(np.mean(series[ref_idx]))

    # --- compute metrics across alphas ---
    means = np.array([np.mean(s) for s in series], dtype=np.float32)
    stds  = np.array([np.std(s, ddof=1) for s in series], dtype=np.float32)
    ns    = np.array([len(s) for s in series], dtype=np.int32)
    errs  = (stds / np.sqrt(ns)) if use_sem else stds

    delta_means = means - mu0  # Δmean(α)

    # proportion above μ0
    prop_above = np.array([(s > mu0).mean() for s in series], dtype=np.float32)

    # --- split negative vs positive alphas for simple trend tests ---
    neg_mask = alpha_vals < 0
    pos_mask = alpha_vals > 0

    def safe_linreg(x, y):
        if np.sum(~np.isnan(x)) >= 2 and np.unique(x).size >= 2:
            r = linregress(x, y)
            return r.slope, r.intercept, r.rvalue, r.pvalue
        return np.nan, np.nan, np.nan, np.nan

    # Δmean trends
    slope_n_dm, b_n_dm, r_n_dm, p_n_dm = safe_linreg(alpha_vals[neg_mask], delta_means[neg_mask])
    slope_p_dm, b_p_dm, r_p_dm, p_p_dm = safe_linreg(alpha_vals[pos_mask], delta_means[pos_mask])

    # proportion trends
    slope_n_pa, b_n_pa, r_n_pa, p_n_pa = safe_linreg(alpha_vals[neg_mask], prop_above[neg_mask])
    slope_p_pa, b_p_pa, r_p_pa, p_p_pa = safe_linreg(alpha_vals[pos_mask], prop_above[pos_mask])

    # --- plotting ---
    setup_pub_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1, ax2 = axes

    # Row 1: Δmean
    ax1.axhline(0.0, linestyle=":", linewidth=1, color="gray")
    ax1.errorbar(alpha_vals, delta_means, yerr=errs, fmt="o-", lw=1.5, ms=4, capsize=3, label="Δmean(α)")
    # overlay separate trend fits (neg & pos) for clarity
    for mask, slope, b, lbl in [
        (neg_mask, slope_n_dm, b_n_dm, f"trend α<0 (s={slope_n_dm:.3g}, p={p_n_dm:.2g})"),
        (pos_mask, slope_p_dm, b_p_dm, f"trend α>0 (s={slope_p_dm:.3g}, p={p_p_dm:.2g})"),
    ]:
        if np.any(mask) and not np.isnan(slope):
            xs = np.linspace(alpha_vals[mask].min(), alpha_vals[mask].max(), 100)
            ax1.plot(xs, slope*xs + b, lw=1.2, label=lbl)
    ax1.set_ylabel("Δmean score vs α=0")
    ref_lab = "original" if used_original else "α=0"
    subj = data.get("subject", None)
    n_total = int(ns.max()) if ns.size else 0
    ax1.set_title(f"{assessor.capitalize()} — Effects vs {ref_lab} (subj{subj:02d}, N≈{n_total})")

    # Row 2: proportion above μ0
    ax2.axhline(0.5, linestyle=":", linewidth=1, color="gray")
    ax2.plot(alpha_vals, prop_above, "o-", lw=1.5, ms=4, label=f"P(score@α > μ_{ref_lab})")
    for mask, slope, b, lbl in [
        (neg_mask, slope_n_pa, b_n_pa, f"trend α<0 (s={slope_n_pa:.3g}, p={p_n_pa:.2g})"),
        (pos_mask, slope_p_pa, b_p_pa, f"trend α>0 (s={slope_p_pa:.3g}, p={p_p_pa:.2g})"),
    ]:
        if np.any(mask) and not np.isnan(slope):
            xs = np.linspace(alpha_vals[mask].min(), alpha_vals[mask].max(), 100)
            ax2.plot(xs, slope*xs + b, lw=1.2, label=lbl)
    ax2.set_xlabel("α")
    ax2.set_ylabel(f"Proportion > μ_{ref_lab}")
    ax2.set_ylim(-0.02, 1.02)

    # Legends
    ax1.legend(loc="best", frameon=True)
    ax2.legend(loc="best", frameon=True)

    fig.tight_layout()
    out_path = out_root / f"{assessor}_theta_alpha_effects_vs_alpha0_{filename_stub}.png"
    save_tight(fig, out_path, dpi=300)
    return out_path


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

    # regression line
    counts_str = ""
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, deg=1)  # slope, intercept
        xx = np.linspace(lims[0], lims[1], 200)
        yy = coeffs[0] * xx + coeffs[1]
        ax.plot(xx, yy, linewidth=1.5, color="C0")

        # --- count above/below regression line ---
        y_pred = coeffs[0] * x + coeffs[1]
        above = np.sum(y > y_pred)
        below = np.sum(y < y_pred)
        counts_str = f"\n↑ {above}, ↓ {below}"

    # y=x reference
    ax.plot(lims, lims, linestyle=":", linewidth=1, color="gray")

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(title + counts_str)
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

        print(f"[{assessor}] Building alpha effects vs alpha0 figure…")
        eff_path = figure_alpha_effects_vs_alpha0(
            data=data,
            assessor=assessor,
            out_root=out_root,
            filename_stub=stub,
            limit=args.limit,
            use_sem=True,   # set False to show SD instead of SEM
        )
        print(f"[{assessor}] Saved: {eff_path}")
        
    print("[DONE] All plots written to:", out_root)




if __name__ == "__main__":
    main()
