#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16_evaluate_theta_sweep.py

Evaluate the effect of applying theta shifts (from 15_shift_and_decode_vdvae.py).
Re-score shifted decodes with EmoNet and MemNet, plot score vs alpha,
and save sample grids for visual inspection.

Expected folder layout (subj01 example):
  /home/rothermm/THINGS/03_results/vdvae_shifted/subj01/{emonet|memnet}/alpha_{±a}/{i}.png

Outputs:
  /home/rothermm/THINGS/03_results/plots/theta_eval/subj01/
    - score_vs_alpha.png
    - samples_alpha{±a}.png
"""

import argparse, sys, pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True); return p

def build_transform(is_memnet=False, mem_mean=None):
    if not is_memnet:
        return T.Compose([
            T.Resize((256,256), Image.BILINEAR),
            T.ToTensor(),
        ])
    else:
        if mem_mean is None:
            raise ValueError("mem_mean required for MemNet")
        return T.Compose([
            T.Resize((256,256), Image.BILINEAR),
            T.Lambda(lambda x: np.array(x.convert("RGB"))),
            T.Lambda(lambda x: np.subtract(x[:, :, [2,1,0]], mem_mean)),  # BGR - mean
            T.Lambda(lambda x: x[15:242, 15:242]),                         # 227x227 crop
            T.ToTensor(),
        ])

@torch.no_grad()
def score_images(paths, assessor, transform, device="cpu"):
    scores = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        s = assessor(x).detach().cpu().numpy().reshape(-1)[0]
        scores.append(float(s))
    return scores

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, required=True)
    ap.add_argument("--shifted_root", type=Path, default=None,
                    help="Root of shifted decodes (default THINGS layout)")
    ap.add_argument("--assessors_root", type=Path, default=Path("/home/rothermm/brain-diffuser/assessors"))
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Where to save plots (default: plots/theta_eval/subjXX)")
    ap.add_argument("--alphas", type=str, nargs="+", default=None,
                    help="Which alphas to evaluate (default: auto-discover)")
    ap.add_argument("--n_samples", type=int, default=8,
                    help="Random samples for grids")
    args = ap.parse_args()

    sp = f"{args.sub:02d}"
    if args.shifted_root is None:
        args.shifted_root = Path(f"/home/rothermm/THINGS/03_results/vdvae_shifted/subj{sp}")
    if args.out_dir is None:
        args.out_dir = Path(f"/home/rothermm/THINGS/03_results/plots/theta_eval/subj{sp}")
    ensure_dir(args.out_dir)

    # --- Load assessors
    sys.path.append(str(args.assessors_root))
    import emonet
    from memnet import MemNet
    emo_model, _, _ = emonet.emonet(tencrop=False)
    assessor_emo = emo_model.eval().requires_grad_(False).to("cpu")
    mem_mean = np.load(args.assessors_root / "image_mean.npy")
    assessor_mem = MemNet().eval().requires_grad_(False).to("cpu")

    tfm_emo = build_transform(is_memnet=False)
    tfm_mem = build_transform(is_memnet=True, mem_mean=mem_mean)

    # --- Discover alphas
    if args.alphas is None:
        alphas = []
        for name in ("emonet","memnet"):
            d = args.shifted_root / name
            if not d.exists(): continue
            for subdir in d.iterdir():
                if subdir.is_dir() and subdir.name.startswith("alpha_"):
                    try:
                        a = float(subdir.name.split("alpha_")[1])
                        alphas.append(a)
                    except: pass
        alphas = sorted(set(alphas))
    else:
        alphas = [float(a) for a in args.alphas]

    print("[INFO] Evaluating alphas:", alphas)

    results = {"emonet": [], "memnet": []}

    for assessor_name, assessor, tfm in [("emonet", assessor_emo, tfm_emo),
                                         ("memnet", assessor_mem, tfm_mem)]:
        base = args.shifted_root / assessor_name
        if not base.exists():
            print(f"[WARN] missing {base}")
            continue

        for a in alphas:
            folder = base / f"alpha_{a:+g}"
            if not folder.exists(): continue
            paths = sorted(folder.glob("*.png"))
            if len(paths) == 0: continue

            scores = score_images(paths, assessor, tfm, device="cpu")
            mean_score = float(np.mean(scores))
            results[assessor_name].append((a, mean_score))
            print(f"[{assessor_name}] alpha={a:+g}, mean_score={mean_score:.4f}")

            # Save a grid of n_samples
            sel = np.random.choice(paths, size=min(args.n_samples,len(paths)), replace=False)
            imgs = [T.ToTensor()(Image.open(p).convert("RGB")) for p in sel]
            grid = make_grid(imgs, nrow=len(imgs), normalize=True)
            save_image(grid, args.out_dir / f"samples_{assessor_name}_alpha{a:+g}.png")

    # --- Plot curves
    plt.figure()
    for name, vals in results.items():
        if not vals: continue
        vals = sorted(vals, key=lambda x: x[0])
        xs, ys = zip(*vals)
        plt.plot(xs, ys, marker="o", label=name)
    plt.xlabel("Alpha step")
    plt.ylabel("Mean assessor score")
    plt.title(f"subj{sp} — Score vs Alpha")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "score_vs_alpha.png", dpi=200)
    plt.close()

    print("[DONE] wrote results to", args.out_dir)

if __name__ == "__main__":
    main()
