#!/usr/bin/env python3
import os, sys, argparse, pickle
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

def load_list(p: Path):
    with open(p, "r") as f: return [ln.strip() for ln in f if ln.strip()]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def load_scores(paths, assessor, *, is_memnet=False, mem_mean=None):
    if not is_memnet:
        tfm = T.Compose([T.Resize((256,256)), T.ToTensor()])
    else:
        tfm = T.Compose([
            T.Resize((256,256), Image.BILINEAR),
            T.Lambda(lambda x: np.array(x)),
            T.Lambda(lambda x: np.subtract(x[:, :, [2,1,0]], mem_mean)),
            T.Lambda(lambda x: x[15:242, 15:242]),
            T.ToTensor()
        ])
    out=[]
    with torch.no_grad():
        for p in paths:
            x = tfm(Image.open(p).convert("RGB")).unsqueeze(0)
            s = assessor(x).detach().cpu().numpy()[0][0]
            out.append(float(s))
    return out

def plot_scores(orig, dec, out_png: Path, title: str):
    x = np.arange(len(orig))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.plot(x, orig, label="Original", linewidth=2)
    ax.plot(x, dec,  label="Decoded",  linewidth=2)
    ax.set_title(title, pad=12, fontsize=14, fontweight="bold")
    ax.set_xlabel("Test image index (ordered)")
    ax.set_ylabel("Assessor score")
    ax.grid(True, alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); ensure_dir(out_png.parent); fig.savefig(out_png, dpi=200); plt.close(fig)

def scatter_with_regression(orig, dec, out_png: Path, title: str):
    ensure_dir(out_png.parent)
    r, p = pearsonr(orig, dec)
    plt.figure(figsize=(5.5,5))
    sns.regplot(x=orig, y=dec, scatter_kws={"s": 20, "alpha": 0.7}, line_kws={"color": "red"})
    plt.xlabel("Original scores")
    plt.ylabel("Decoded scores")
    plt.title(f"{title}\nPearson r = {r:.3f}, p = {p:.2e}", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--test_paths", type=Path, default=None)
    ap.add_argument("--decoded_dir", type=Path, default=None)
    ap.add_argument("--assessors_root", type=Path, default=Path("/home/rothermm/brain-diffuser/assessors"))
    ap.add_argument("--scores_dir", type=Path, default=None)
    ap.add_argument("--plots_dir", type=Path, default=Path("/home/rothermm/THINGS/03_results/plots"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sp = f"{args.sub:02d}"
    if args.test_paths is None:
        args.test_paths = Path(f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sp}/test_image_paths.txt")
    if args.decoded_dir is None:
        args.decoded_dir = Path(f"/home/rothermm/THINGS/03_results/vdvae/subj{sp}")
    if args.scores_dir is None:
        args.scores_dir = Path(f"/home/rothermm/THINGS/03_results/assessor_scores/subj{sp}")

    print("[INFO] test_paths:", args.test_paths)
    print("[INFO] decoded_dir:", args.decoded_dir)
    print("[INFO] scores_dir:", args.scores_dir)
    print("[INFO] plots_dir:", args.plots_dir)

    # Existence checks
    if not args.test_paths.exists(): raise FileNotFoundError(f"Missing: {args.test_paths}")
    if not args.decoded_dir.exists(): raise FileNotFoundError(f"Missing: {args.decoded_dir}")
    ensure_dir(args.scores_dir); ensure_dir(args.plots_dir)

    test_paths = load_list(args.test_paths)
    if args.limit is not None: test_paths = test_paths[:args.limit]
    decoded_paths = [str(args.decoded_dir / f"{i}.png") for i in range(len(test_paths))]
    missing = [p for p in decoded_paths if not Path(p).exists()]
    print(f"[INFO] #test={len(test_paths)}  #decoded_expected={len(decoded_paths)}  #decoded_missing={len(missing)}")
    if missing:
        print("[ERR] Missing decoded examples; first few:", missing[:5])
        raise FileNotFoundError("Decoded images missing; aborting.")

    # Load assessors
    sys.path.append(str(args.assessors_root))
    print("[INFO] importing assessors from:", args.assessors_root)
    import emonet
    from memnet import MemNet
    print("[INFO] loading emonet/memnet ...")
    model, _, _ = emonet.emonet(tencrop=False)
    assessor_emo = model.eval().requires_grad_(False).to("cpu")
    mem_mean = np.load(args.assessors_root / "image_mean.npy")
    assessor_mem = MemNet().eval().requires_grad_(False).to("cpu")

    # Score & save — Emonet
    print("[INFO] scoring Emonet ...")
    emo_original = load_scores(test_paths, assessor_emo, is_memnet=False)
    emo_decoded  = load_scores(decoded_paths, assessor_emo, is_memnet=False)
    with open(args.scores_dir / f"emonet_scores_sub{sp}.pkl", "wb") as f:
        pickle.dump({"original": emo_original, "decoded": emo_decoded}, f)
    print("[INFO] saved emonet scores.")

    """
    plot_scores(
        emo_original, emo_decoded,
        args.plots_dir / f"emonet_original_vs_decoded_sub{sp}.png",
        f"EmoNet Scores — Original vs Decoded (subj{sp})"
    )
    """

    # Scatter plot for EmoNet
    scatter_with_regression(
        emo_original, emo_decoded,
        args.plots_dir / f"emonet_scatter_sub{sp}.png",
        f"EmoNet Original vs Decoded (subj{sp})"
    )

    # Score & save — MemNet
    print("[INFO] scoring MemNet ...")
    mem_original = load_scores(test_paths, assessor_mem, is_memnet=True, mem_mean=mem_mean)
    mem_decoded  = load_scores(decoded_paths, assessor_mem, is_memnet=True, mem_mean=mem_mean)
    with open(args.scores_dir / f"memnet_scores_sub{sp}.pkl", "wb") as f:
        pickle.dump({"original": mem_original, "decoded": mem_decoded}, f)
    print("[INFO] saved memnet scores.")

    """
    plot_scores(
        mem_original, mem_decoded,
        args.plots_dir / f"memnet_original_vs_decoded_sub{sp}.png",
        f"MemNet Scores — Original vs Decoded (subj{sp})"
    )
    """
    
    # Scatter plot for MemNet
    scatter_with_regression(
        mem_original, mem_decoded,
        args.plots_dir / f"memnet_scatter_sub{sp}.png",
        f"MemNet Original vs Decoded (subj{sp})"
    )
    
    print("[DONE] Wrote plots to:", args.plots_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback as tb
        tb.print_exc()
        sys.exit(1)
