#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, pickle
from pathlib import Path
from typing import Tuple  # <-- Py3.8-compatible
import numpy as np

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def get_top_bottom_idx(scores: np.ndarray, frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of top and bottom fractions (at least 1 each)."""
    n = len(scores)
    k = max(1, int(round(n * frac)))
    order = np.argsort(scores)          # ascending
    bottom = order[:k]
    top = order[-k:]
    return top, bottom

def compute_theta(scores: np.ndarray, test_latents: np.ndarray, frac: float):
    assert len(scores) == test_latents.shape[0], \
        f"scores ({len(scores)}) and test_latents ({test_latents.shape[0]}) length mismatch"
    top_idx, bot_idx = get_top_bottom_idx(scores, frac)
    z_top = test_latents[top_idx].mean(axis=0)
    z_bot = test_latents[bot_idx].mean(axis=0)
    theta = z_top - z_bot
    return {
        "theta": theta,
        "top_idx": top_idx,
        "bot_idx": bot_idx,
        "top_mean_score": float(scores[top_idx].mean()),
        "bot_mean_score": float(scores[bot_idx].mean()),
    }

def main():
    ap = argparse.ArgumentParser(description="Compute theta from saved assessor scores (THINGS).")
    ap.add_argument("--sub", type=int, default=1, help="Subject number (e.g., 1)")
    ap.add_argument("--scores_dir", type=Path, default=None,
                    help="Folder with emonet_scores_subXX.pkl, memnet_scores_subXX.pkl")
    ap.add_argument("--features_npz", type=Path, default=None,
                    help="NPZ containing test_latents")
    ap.add_argument("--which", choices=["decoded", "original"], default="decoded",
                    help="Which score stream to use (default: decoded)")
    ap.add_argument("--frac", type=float, default=0.10,
                    help="Top/bottom fraction to use (default: 0.10)")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Where to save theta vectors and summary JSON")
    ap.add_argument("--normalize", action="store_true",
                    help="If set, L2-normalize theta before saving")
    args = ap.parse_args()

    sp = f"{args.sub:02d}"
    if args.scores_dir is None:
        args.scores_dir = Path(f"/home/rothermm/THINGS/03_results/assessor_scores/subj{sp}")
    if args.features_npz is None:
        args.features_npz = Path(f"/home/rothermm/THINGS/02_data/extracted_features/subj{sp}/things_vdvae_features_31l.npz")
    if args.out_dir is None:
        args.out_dir = Path(f"/home/rothermm/THINGS/03_results/thetas/subj{sp}")
    ensure_dir(args.out_dir)

    emo_pkl = args.scores_dir / f"emonet_scores_sub{sp}.pkl"
    mem_pkl = args.scores_dir / f"memnet_scores_sub{sp}.pkl"
    for p in (emo_pkl, mem_pkl, args.features_npz):
        if not p.exists(): raise FileNotFoundError(f"Missing: {p}")

    with open(emo_pkl, "rb") as f: emo = pickle.load(f)
    with open(mem_pkl, "rb") as f: mem = pickle.load(f)

    emo_scores = np.asarray(emo[args.which], dtype=float)
    mem_scores = np.asarray(mem[args.which], dtype=float)

    ft = np.load(args.features_npz)
    if "test_latents" not in ft:
        raise KeyError(f"'test_latents' not found in {args.features_npz}")
    Z_test = ft["test_latents"]

    if not (len(emo_scores) == len(mem_scores) == Z_test.shape[0]):
        raise ValueError(f"Length mismatch: emonet={len(emo_scores)} memnet={len(mem_scores)} test_latents={Z_test.shape[0]}")

    emo_res = compute_theta(emo_scores, Z_test, args.frac)
    mem_res = compute_theta(mem_scores, Z_test, args.frac)

    theta_emo = emo_res["theta"]
    theta_mem = mem_res["theta"]

    if args.normalize:
        def l2(x): 
            n = np.linalg.norm(x) + 1e-12
            return x / n
        theta_emo = l2(theta_emo)
        theta_mem = l2(theta_mem)

    tag = f"{args.which}_top{int(args.frac*100)}_minus_bottom{int(args.frac*100)}"
    emo_out = args.out_dir / f"theta_emonet_{tag}.npy"
    mem_out = args.out_dir / f"theta_memnet_{tag}.npy"
    np.save(emo_out, theta_emo)
    np.save(mem_out, theta_mem)

    summary = {
        "subject": args.sub,
        "which_scores": args.which,
        "frac": args.frac,
        "normalize": bool(args.normalize),
        "n_test": int(Z_test.shape[0]),
        "emonet": {
            "out": str(emo_out),
            "top_mean_score": emo_res["top_mean_score"],
            "bot_mean_score": mem_res["bot_mean_score"],
            "top_idx_sample": emo_res["top_idx"][:20].tolist(),
            "bot_idx_sample": emo_res["bot_idx"][:20].tolist(),
        },
        "memnet": {
            "out": str(mem_out),
            "top_mean_score": mem_res["top_mean_score"],
            "bot_mean_score": mem_res["bot_mean_score"],
            "top_idx_sample": mem_res["top_idx"][:20].tolist(),
            "bot_idx_sample": mem_res["bot_idx"][:20].tolist(),
        },
    }
    with open(args.out_dir / f"theta_summary_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE]")
    print("  EmoNet theta ->", emo_out)
    print("  MemNet theta ->", mem_out)
    print("  Summary       ->", args.out_dir / f"theta_summary_{tag}.json")

if __name__ == "__main__":
    main()
