#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
17_assessor_scores_theta.py  (no per-alpha plots)

Compute assessor scores for original images and all shifted-decode alpha folders,
and save structured results (scores, means, stds, correlations) per assessor.

Outputs (per assessor) one pickle in:
  /home/rothermm/THINGS/03_results/assessor_scores/subjXX/theta/
    - emonet_theta_scores_subXX.pkl
    - memnet_theta_scores_subXX.pkl

Each pickle contains:
  - "scores": {"original", "alpha_{±a}"} -> score lists
  - "means" / "stds": summary stats (float)
  - "correlations": {"alpha_{±a}": {"r": ..., "p": ...}}
  - "paths": minimal provenance info

Assumes shifted decodes are at:
  /home/rothermm/THINGS/03_results/vdvae_shifted/subjXX/{emonet|memnet}/alpha_{±a}/{i}.png
"""

import os
import re
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from scipy.stats import pearsonr

# -------------------- utils --------------------

def load_list(p: Path):
    with open(p, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_alpha(dirname: str) -> float:
    m = re.match(r"^alpha_([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?)$", dirname, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unrecognized alpha dir name: {dirname}")
    return float(m.group(1))

def alpha_key(a: float) -> str:
    return f"alpha_{a:+g}"

def expected_indexed_pngs(folder: Path, n: int):
    exp = [folder / f"{i}.png" for i in range(n)]
    if all(p.exists() for p in exp):
        return exp
    files = sorted(folder.glob("*.png"), key=lambda p: int(Path(p).stem))
    if len(files) != n:
        missing = [str(p) for p in exp if not p.exists()][:5]
        raise FileNotFoundError(f"{folder}: expected {n} images; found {len(files)}. Missing (first 5): {missing}")
    return files

def build_transform(is_memnet: bool, mem_mean=None):
    if not is_memnet:
        return T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),  # -> [0,1]
        ])
    else:
        if mem_mean is None:
            raise ValueError("mem_mean is required for MemNet preprocessing")
        return T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: np.array(x.convert("RGB"))),                # HWC, RGB uint8
            T.Lambda(lambda x: np.subtract(x[:, :, [2, 1, 0]], mem_mean)), # BGR - mean (pre-ToTensor scale)
            T.Lambda(lambda x: x[15:242, 15:242]),                         # 227x227 center crop
            T.ToTensor(),                                                  # -> [0,1] after mean subtraction
        ])

@torch.no_grad()
def score_image_batch(paths, assessor, transform, device="cpu", bs=64):
    scores = np.empty(len(paths), dtype=np.float32)
    i = 0
    while i < len(paths):
        chunk = paths[i:i+bs]
        tensors = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
        x = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        y = assessor(x)                        # shape: [B, 1] or [B]
        y = y.detach().view(-1).cpu().numpy()
        scores[i:i+len(chunk)] = y
        i += len(chunk)
    return scores.tolist()

def load_assessors(assessors_root: Path, device: str):
    sys.path.append(str(assessors_root))
    import emonet
    from memnet import MemNet

    model, _, _ = emonet.emonet(tencrop=False)
    assessor_emo = model.eval().requires_grad_(False).to(device)

    mem_mean = np.load(assessors_root / "image_mean.npy")
    assessor_mem = MemNet().eval().requires_grad_(False).to(device)
    return assessor_emo, assessor_mem, mem_mean

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=1, help="Subject number, e.g., 1")
    ap.add_argument("--test_paths", type=Path, default=None,
                    help="File with original test image paths (defaults to THINGS layout)")
    ap.add_argument("--shifted_root", type=Path, default=None,
                    help="Root of shifted decodes (defaults to THINGS layout)")
    ap.add_argument("--assessors_root", type=Path, default=Path("/home/rothermm/brain-diffuser/assessors"))
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Pickle output dir (defaults to .../assessor_scores/subjXX/theta)")
    ap.add_argument("--assessors", choices=["both", "emonet", "memnet"], default="both")
    ap.add_argument("--alphas", type=float, nargs="*", default=None,
                    help="If provided, only score these alphas; else auto-discover from folders")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on N")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    sp = f"{args.sub:02d}"
    if args.test_paths is None:
        args.test_paths = Path(f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sp}/test_image_paths.txt")
    if args.shifted_root is None:
        args.shifted_root = Path(f"/home/rothermm/THINGS/03_results/vdvae_shifted/subj{sp}")
    if args.out_dir is None:
        args.out_dir = Path(f"/home/rothermm/THINGS/03_results/assessor_scores/subj{sp}/theta")
    ensure_dir(args.out_dir)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("[INFO] test_paths   :", args.test_paths)
    print("[INFO] shifted_root :", args.shifted_root)
    print("[INFO] out_dir      :", args.out_dir)
    print("[INFO] device       :", device)

    if not args.test_paths.exists():
        raise FileNotFoundError(f"Missing: {args.test_paths}")
    if not args.shifted_root.exists():
        raise FileNotFoundError(f"Missing: {args.shifted_root}")

    # Load original image list
    test_paths = load_list(args.test_paths)
    if args.limit is not None:
        test_paths = test_paths[:args.limit]
    N = len(test_paths)
    print(f"[INFO] N test images: {N}")

    # Load assessors + transforms
    assessor_emo, assessor_mem, mem_mean = load_assessors(args.assessors_root, device)
    tfm_emo = build_transform(is_memnet=False)
    tfm_mem = build_transform(is_memnet=True, mem_mean=mem_mean)

    def process_family(name: str, assessor, transform):
        base = args.shifted_root / name
        if not base.exists():
            raise FileNotFoundError(f"Missing shifted folder for '{name}': {base}")

        # Discover alpha folders
        alpha_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("alpha_")]
        if not alpha_dirs:
            raise FileNotFoundError(f"No alpha_* folders in {base}")
        alpha_map = {}
        for d in alpha_dirs:
            try:
                a = parse_alpha(d.name)
                alpha_map[a] = d
            except ValueError:
                continue

        # Filter/sort alphas
        if args.alphas is not None and len(args.alphas) > 0:
            wanted = set(float(a) for a in args.alphas)
            alpha_items = sorted([(a, alpha_map[a]) for a in wanted], key=lambda x: x[0])
        else:
            alpha_items = sorted(alpha_map.items(), key=lambda x: x[0])

        # Score ORIGINAL natural images
        print(f"[{name}] Scoring ORIGINAL images...")
        original_scores = score_image_batch(test_paths, assessor, transform, device=device, bs=args.batch_size)

        # Prepare outputs
        series = {"original": original_scores}
        means  = {"original": float(np.mean(original_scores))}
        stds   = {"original": float(np.std(original_scores, ddof=1))}
        cors   = {}   # per-alpha r/p

        # Score each alpha folder
        for a, d in alpha_items:
            print(f"[{name}] Scoring {d.name} ...")
            alpha_paths = expected_indexed_pngs(d, N)
            s = score_image_batch(alpha_paths, assessor, transform, device=device, bs=args.batch_size)
            key = alpha_key(a)
            series[key] = s
            means[key] = float(np.mean(s))
            stds[key]  = float(np.std(s, ddof=1))

            # Correlation vs original (no plotting)
            r, p = pearsonr(original_scores, s)
            cors[key] = {"r": float(r), "p": float(p)}

        # Persist results
        out = {
            "subject": int(args.sub),
            "assessor": name,
            "n": int(N),
            "alphas": [float(a) for a, _ in alpha_items],
            "scores": series,
            "means": means,
            "stds": stds,
            "correlations": cors,
            "paths": {
                "test_paths_file": str(args.test_paths),
                "shifted_root": str(base),
            },
        }
        out_pkl = args.out_dir / f"{name}_theta_scores_sub{sp}.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(out, f)
        print(f"[{name}] Saved -> {out_pkl}")

    # Run selected assessors
    if args.assessors in ("both", "emonet"):
        process_family("emonet", assessor_emo, tfm_emo)
    if args.assessors in ("both", "memnet"):
        process_family("memnet", assessor_mem, tfm_mem)

    print("[DONE] All theta score files written to:", args.out_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback as tb
        tb.print_exc()
        sys.exit(1)
