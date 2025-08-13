#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_compare_original_vs_decoded.py

Create a side-by-side comparison figure of ORIGINAL vs DECODED images
for the first N examples (default: 10) of the *test* set.

Assumptions:
- test_image_paths.txt lists absolute paths to the N test images, in order
  matching your predicted/decoded outputs.
- Decoded images live in OUT_DIR and are named '0.png', '1.png', ..., in order.
- Works out-of-the-box with the THINGS paths from previous steps.

Example:
python 09_compare_original_vs_decoded.py \
  --sub 1 \
  --test_paths /home/rothermm/THINGS/02_data/preprocessed_data/subj01/test_image_paths.txt \
  --decoded_dir /home/rothermm/THINGS/03_results/vdvae/subj01 \
  --out_png /home/rothermm/THINGS/03_results/vdvae/subj01/compare_first10.png \
  --n 10
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def load_paths(txt_path: Path):
    with open(txt_path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def smart_open_rgb(p: Path, target_hw=None):
    img = Image.open(p).convert("RGB")
    if target_hw is not None:
        img = img.resize((target_hw[1], target_hw[0]), resample=Image.BICUBIC)
    return np.array(img)

def main():
    ap = argparse.ArgumentParser(description="Compare original vs decoded images for first N test items.")
    ap.add_argument("--sub", type=int, default=1, help="Subject number (zero-padded in paths).")
    ap.add_argument("--test_paths", type=Path, default=None, help="test_image_paths.txt")
    ap.add_argument("--decoded_dir", type=Path, default=None, help="Directory with decoded images named 0.png,1.png,...")
    ap.add_argument("--out_png", type=Path, default=None, help="Output figure PNG path.")
    ap.add_argument("--n", type=int, default=10, help="How many examples to show.")
    ap.add_argument("--img_size", type=int, default=256, help="Side length to display (both original/decoded).")
    args = ap.parse_args()

    subj_pad = f"{args.sub:02d}"

    # Defaults for THINGS layout
    if args.test_paths is None:
        args.test_paths = Path(f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{subj_pad}/test_image_paths.txt")
    if args.decoded_dir is None:
        args.decoded_dir = Path(f"/home/rothermm/THINGS/03_results/vdvae/subj{subj_pad}")
    if args.out_png is None:
        args.out_png = Path(f"/home/rothermm/THINGS/03_results/plots/compare_first{args.n}.png")

    # Load list of originals
    test_paths = load_paths(args.test_paths)
    if len(test_paths) == 0:
        raise RuntimeError(f"No paths in {args.test_paths}")
    n = min(args.n, len(test_paths))

    # Preload originals and decoded (resize both to same height/width)
    target_hw = (args.img_size, args.img_size)
    originals = []
    decodeds = []
    names = []

    for i in range(n):
        orig_p = Path(test_paths[i])
        dec_p = args.decoded_dir / f"{i}.png"  # relies on decoded order
        if not orig_p.exists():
            raise FileNotFoundError(f"Missing original: {orig_p}")
        if not dec_p.exists():
            raise FileNotFoundError(f"Missing decoded: {dec_p}")

        orig_img = smart_open_rgb(orig_p, target_hw)
        dec_img  = smart_open_rgb(dec_p, target_hw)

        originals.append(orig_img)
        decodeds.append(dec_img)
        names.append(orig_p.name)

    # Build figure: 2 columns (Original | Decoded) x n rows
    # Aesthetic: large figure, subtle titles, no axes, tight layout
    fig_h = n * 2.6  # vertical scaling per row
    fig_w = 2 * 2.6  # two columns
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(fig_w, fig_h))
    if n == 1:
        axes = np.array([axes])  # make it 2D for consistency

    # Titles for top row
    axes[0, 0].set_title("Original", fontsize=14, pad=10, weight="bold")
    axes[0, 1].set_title("Decoded (VDVAE)", fontsize=14, pad=10, weight="bold")

    for i in range(n):
        # Left: original
        axL = axes[i, 0]
        axL.imshow(originals[i])
        axL.set_ylabel(f"{i:02d}  {names[i]}", fontsize=10)
        axL.set_xticks([]); axL.set_yticks([])
        for spine in axL.spines.values():
            spine.set_visible(False)

        # Right: decoded
        axR = axes[i, 1]
        axR.imshow(decodeds[i])
        axR.set_xticks([]); axR.set_yticks([])
        for spine in axR.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=200)
    print(f"[DONE] Wrote comparison figure -> {args.out_png}")

if __name__ == "__main__":
    main()
