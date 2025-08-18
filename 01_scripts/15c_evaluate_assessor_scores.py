#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-image effect of theta: slope of assessor score vs alpha, plus change-in-pixels sanity.

Reads shifted decodes from:
  /home/rothermm/THINGS/03_results/vdvae_shifted/subjXX/{emonet|memnet}/alpha_{±a}/{i}.png

Outputs:
  /home/rothermm/THINGS/03_results/plots/theta_eval/subjXX/
    - slopes_hist_{assessor}.png
    - mean_score_vs_alpha.png
    - mean_abs_pixel_delta_vs_alpha_{assessor}.png
    - slopes_{assessor}.csv
"""

import argparse, sys, csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def build_transform(is_memnet=False, mem_mean=None):
    if not is_memnet:
        return T.Compose([T.Resize((256,256), Image.BILINEAR), T.ToTensor()])
    if mem_mean is None: raise ValueError("mem_mean required for MemNet")
    return T.Compose([
        T.Resize((256,256), Image.BILINEAR),
        T.Lambda(lambda x: np.array(x.convert("RGB"))),
        T.Lambda(lambda x: np.subtract(x[:, :, [2,1,0]], mem_mean)),  # BGR-mean
        T.Lambda(lambda x: x[15:242, 15:242]),                         # 227x227 crop
        T.ToTensor()
    ])

@torch.no_grad()
def score_paths(paths, assessor, tfm, device="cpu"):
    out=[]; 
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        s = assessor(x).detach().cpu().numpy().reshape(-1)[0]
        out.append(float(s))
    return np.asarray(out, dtype=np.float32)

def list_alpha_dirs(root: Path):
    alphas=[]
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("alpha_"):
            try: alphas.append((float(d.name.split("alpha_")[1]), d))
            except: pass
    return [d for _,d in sorted(alphas, key=lambda t: t[0])]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, required=True)
    ap.add_argument("--shifted_root", type=Path, default=None)
    ap.add_argument("--assessors_root", type=Path, default=Path("/home/rothermm/brain-diffuser/assessors"))
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--max_imgs", type=int, default=None, help="optional limit for speed")
    args = ap.parse_args()

    sp = f"{args.sub:02d}"
    shifted_root = args.shifted_root or Path(f"/home/rothermm/THINGS/03_results/vdvae_shifted/subj{sp}")
    out_dir = args.out_dir or Path(f"/home/rothermm/THINGS/03_results/plots/theta_eval/subj{sp}")
    ensure_dir(out_dir)

    # Assessors
    sys.path.append(str(args.assessors_root))
    import emonet
    from memnet import MemNet
    emo_model, _, _ = emonet.emonet(tencrop=False)
    assessor_emo = emo_model.eval().requires_grad_(False).to("cpu")
    mem_mean = np.load(args.assessors_root / "image_mean.npy")
    assessor_mem = MemNet().eval().requires_grad_(False).to("cpu")
    tfm_emo = build_transform(False)
    tfm_mem = build_transform(True, mem_mean)

    results_curve = {}  # assessor -> (alphas, mean_scores)

    for name, assessor, tfm in [("emonet", assessor_emo, tfm_emo),
                                ("memnet", assessor_mem, tfm_mem)]:
        base = shifted_root / name
        if not base.exists():
            print(f"[WARN] missing {base}"); continue
        a_dirs = list_alpha_dirs(base)
        if not a_dirs:
            print(f"[WARN] no alpha_* in {base}"); continue

        # build a consistent index set present in ALL alpha folders
        sets = []
        for d in a_dirs:
            idx = [int(p.stem) for p in d.glob("*.png") if p.stem.isdigit()]
            sets.append(set(idx))
        common_idx = sorted(list(set.intersection(*sets))) if sets else []
        if args.max_imgs: common_idx = common_idx[:args.max_imgs]
        if not common_idx:
            print(f"[WARN] no common indices across alphas for {name}"); continue

        # score per alpha (aligned by index)
        alphas = [float(d.name.split("alpha_")[1]) for d in a_dirs]
        scores_per_alpha = []
        mean_abs_pix_delta = []  # vs alpha=0
        # cache alpha=0 RGBs for pixel delta
        a0_dir = {d.name: d for d in a_dirs}.get("alpha_+0", None)

        for d in a_dirs:
            paths = [d / f"{i}.png" for i in common_idx]
            s = score_paths(paths, assessor, tfm, device="cpu")
            scores_per_alpha.append(s)

            # pixel delta vs alpha 0 (if available)
            if a0_dir is not None:
                mdiffs=[]
                for i in common_idx:
                    try:
                        A = np.asarray(Image.open(d / f"{i}.png").convert("RGB"), dtype=np.float32)
                        B = np.asarray(Image.open(a0_dir / f"{i}.png").convert("RGB"), dtype=np.float32)
                        mdiffs.append(np.mean(np.abs(A-B)))
                    except: pass
                mean_abs_pix_delta.append(float(np.mean(mdiffs)) if mdiffs else np.nan)
            else:
                mean_abs_pix_delta.append(np.nan)

        S = np.stack(scores_per_alpha, axis=1)  # [Nimg, Nalpha]
        # per-image slope via least squares: slope = cov(a, s)/var(a)
        a = np.asarray(alphas, dtype=np.float32)
        a = (a - a.mean()) / (a.std() + 1e-12)
        slopes = (S @ a) / (len(a) - 1)  # proportional to covariance with standardized alpha

        # save histogram of slopes
        plt.figure()
        sns.histplot(slopes, bins=40)
        plt.axvline(0, color="k", linestyle="--", linewidth=1)
        plt.title(f"subj{sp} {name}: per-image slope of score vs alpha\n(+ means score increases with +alpha)")
        plt.xlabel("slope")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(out_dir / f"slopes_hist_{name}.png", dpi=200)
        plt.close()

        # save CSV
        with open(out_dir / f"slopes_{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["image_index","slope"])
            for idx, sl in zip(common_idx, slopes): w.writerow([idx, float(sl)])

        # mean score vs alpha curve
        mean_scores = [float(np.mean(s)) for s in scores_per_alpha]
        results_curve[name] = (alphas, mean_scores)

        # plot pixel delta vs alpha
        plt.figure()
        plt.plot(alphas, mean_abs_pix_delta, marker="o")
        plt.xlabel("alpha"); plt.ylabel("mean |Δpixel| vs α=0")
        plt.title(f"subj{sp} {name}: pixel change vs alpha")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(out_dir / f"mean_abs_pixel_delta_vs_alpha_{name}.png", dpi=200)
        plt.close()

    # joint curve
    plt.figure()
    for name, (xs, ys) in results_curve.items():
        xs2, ys2 = zip(*sorted(zip(xs, ys)))
        plt.plot(xs2, ys2, marker="o", label=name)
    plt.xlabel("Alpha step")
    plt.ylabel("Mean assessor score")
    plt.title(f"subj{sp} — Mean score vs alpha")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "mean_score_vs_alpha.png", dpi=200)
    plt.close()

    print("[DONE] Wrote:", out_dir)

if __name__ == "__main__":
    main()
