#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_plot_shifted_examples.py

Make a panel comparing shifted decodes across alphas for both 'emonet' and 'memnet'.
Rows = examples (first N) for each assessor; Columns = alphas (e.g., -4, -3, -2, 0, 2, 3, 4).

Inputs (defaults for subj01):
  /home/rothermm/THINGS/03_results/vdvae_shifted/subj01/{emonet|memnet}/alpha_{±k}/{idx}.png

Output:
  /home/rothermm/THINGS/03_results/plots/alpha_grid_sub01.png
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def fmt_alpha_dir(a: float) -> str:
    """alpha directory name using signed compact format, e.g. alpha_-4, alpha_+0, alpha_+0.5"""
    return f"alpha_{a:+g}"

def load_img(path: Path, size: int = 256):
    """Return HWC uint8 array or None if missing."""
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=1, help="Subject number")
    ap.add_argument("--shifted_base", type=Path, default=None, help="Base dir of shifted decodes")
    ap.add_argument("--out_png", type=Path, default=None, help="Exact output PNG path")
    ap.add_argument("--plots_dir", type=Path, default=None,
                    help="If given, save to <plots_dir>/alpha_grid_subXX.png")
    ap.add_argument("--num_examples", type=int, default=4, help="How many examples per assessor")
    ap.add_argument("--n_examples", type=int, default=None, help="Alias of --num_examples")
    ap.add_argument("--img_size", type=int, default=256, help="Tile size in pixels")
    ap.add_argument("--tile", type=int, default=None, help="Alias of --img_size")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[-4, -3, -2, 0, 2, 3, 4],
                    help="Alpha values shown left→right")
    args = ap.parse_args()

    # Back-compat aliases
    if args.n_examples is not None:
        args.num_examples = args.n_examples
    if args.tile is not None:
        args.img_size = args.tile

    sp = f"{args.sub:02d}"
    if args.shifted_base is None:
        args.shifted_base = Path(f"/home/rothermm/THINGS/03_results/vdvae_shifted/subj{sp}")

    # Output path selection
    if args.plots_dir is not None:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        args.out_png = args.plots_dir / f"alpha_grid_sub{sp}.png"
    if args.out_png is None:
        args.out_png = Path(f"/home/rothermm/THINGS/03_results/plots/alpha_grid_sub{sp}.png")
        args.out_png.parent.mkdir(parents=True, exist_ok=True)

    assessors = ["emonet", "memnet"]
    alpha_dirs = [fmt_alpha_dir(a) for a in args.alphas]

    # Sanity check directories
    for a_name in assessors:
        for ad in alpha_dirs:
            d = args.shifted_base / a_name / ad
            if not d.exists():
                raise FileNotFoundError(f"Missing directory: {d}")

    # Build figure: rows = len(assessors) * num_examples, cols = len(alphas)
    rows = len(assessors) * args.num_examples
    cols = len(alpha_dirs)
    fig_w = 2.4 * cols
    fig_h = 2.4 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    axes = axes.reshape(rows, cols)

    for ai, a_name in enumerate(assessors):
        for ex in range(args.num_examples):
            row = ai * args.num_examples + ex
            for ci, ad in enumerate(alpha_dirs):
                img_path = args.shifted_base / a_name / ad / f"{ex}.png"
                arr = load_img(img_path, size=args.img_size)
                ax = axes[row, ci]
                ax.axis("off")
                if arr is None:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
                else:
                    ax.imshow(arr)
                if ex == 0:
                    # Column headers only on top row of each assessor block
                    ax.set_title(ad.replace("alpha_", "α="), fontsize=10, pad=6)
            # Row label at far-left
            axes[row, 0].set_ylabel(f"{a_name}\nex#{ex}", fontsize=10)

    fig.suptitle(f"Shifted VDVAE decodes across α — subj {sp}", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(args.out_png, dpi=200)
    print(f"[DONE] Saved: {args.out_png}")

if __name__ == "__main__":
    main()
