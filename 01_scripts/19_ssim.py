#!/usr/bin/env python3
"""
Compute SSIM for VDVAE theta manipulations and produce barplots.

- Shifted images: /home/rothermm/THINGS/03_results/vdvae_shifted/subjXX/{emonet|memnet}/alpha_{+/-K[.5]}/
- Originals root list for the subject (absolute paths, one per line):
    /home/rothermm/THINGS/02_data/preprocessed_data/subjXX/test_image_paths.txt

This version AUTO-DETECTS original-image matching strategy using alpha_+0:
  strategy 'name' : shifted stem == original stem
  strategy 'idx0' : shifted stem is 0-based index into test_image_paths.txt
  strategy 'idx1' : shifted stem is 1-based index into test_image_paths.txt

Outputs:
  /home/rothermm/THINGS/03_results/plots/ssim/
    - subjXX_{model}_ssim_per_image.csv
    - subjXX_{model}_mean_ssim_vs_alpha0.png
    - subjXX_{model}_mean_ssim_vs_original.png
    - subjXX_{model}_summary.json
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def load_image_gray_float(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def compute_ssim_pair(a: Path, b: Path) -> float:
    A = load_image_gray_float(a)
    B = load_image_gray_float(b)
    if A.shape != B.shape:
        with Image.open(b) as im:
            im = im.convert("L").resize((A.shape[1], A.shape[0]), Image.BILINEAR)
            B = np.asarray(im, dtype=np.float32) / 255.0
    return float(ssim(A, B, data_range=1.0, gaussian_weights=True, use_sample_covariance=False))

def parse_alpha_dirname(dname: str) -> Optional[float]:
    m = re.fullmatch(r"alpha_([+-]?\d+(?:\.\d+)?)", dname)
    return float(m.group(1)) if m else None

def list_images(d: Path) -> List[Path]:
    return [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]

def read_original_list(stim_paths_file: Path) -> List[Path]:
    out = []
    with stim_paths_file.open("r") as f:
        for line in f:
            sp = line.strip()
            if sp:
                out.append(Path(sp))
    return out

def build_name_map(originals: List[Path]) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for p in originals:
        m.setdefault(p.stem, p)  # first occurrence wins
    return m

def choose_mapping_strategy(alpha0_dir: Path, originals: List[Path]) -> Tuple[str, Dict[str, Path]]:
    """
    Look at stems in alpha_+0 and see which mapping yields most hits:
      - name  : stem matches original stem
      - idx0  : int(stem) used as 0-based index into originals
      - idx1  : int(stem) used as 1-based index into originals
    Returns (strategy_name, mapping_dict_for_that_strategy where key is shifted stem).
    """
    stems = [p.stem for p in list_images(alpha0_dir)]
    name_map = build_name_map(originals)

    # Build index maps (as strings)
    idx0_map: Dict[str, Path] = {str(i): originals[i] for i in range(len(originals))}
    idx1_map: Dict[str, Path] = {str(i + 1): originals[i] for i in range(len(originals))}

    def count_hits(mapping: Dict[str, Path]) -> int:
        return sum(1 for s in stems if s in mapping)

    hits_name = count_hits(name_map)
    hits_idx0 = count_hits(idx0_map)
    hits_idx1 = count_hits(idx1_map)

    best = max([("name", hits_name), ("idx0", hits_idx0), ("idx1", hits_idx1)], key=lambda x: x[1])

    if best[0] == "name":
        chosen_map = name_map
    elif best[0] == "idx0":
        chosen_map = idx0_map
    else:
        chosen_map = idx1_map

    print(f"[match] alpha_0 probes → name={hits_name}, idx0={hits_idx0}, idx1={hits_idx1} → choosing '{best[0]}'")

    return best[0], chosen_map

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def compute_for_model(
    subject_str: str,
    model: str,
    shifted_base: Path,
    stim_paths_file: Path,
    plots_dir: Path,
    selected_alphas: Optional[List[float]],
    min_n: int = 5,
) -> Dict[str, str]:
    model_root = shifted_base / model
    if not model_root.is_dir():
        raise FileNotFoundError(f"Missing directory: {model_root}")

    # Collect alpha dirs
    alpha_dirs: Dict[float, Path] = {}
    for d in sorted(model_root.iterdir()):
        if not d.is_dir():
            continue
        a = parse_alpha_dirname(d.name)
        if a is not None:
            alpha_dirs[a] = d
    if 0.0 not in alpha_dirs and +0.0 not in alpha_dirs:
        raise FileNotFoundError(f"[{model}] Could not find alpha_+0 under {model_root}")
    alpha0_dir = alpha_dirs.get(0.0, alpha_dirs.get(+0.0))

    # Which alphas to use
    all_avail = sorted(alpha_dirs.keys())
    alphas = all_avail if selected_alphas is None else [a for a in selected_alphas if a in alpha_dirs]

    # Load originals list and choose strategy
    originals = read_original_list(stim_paths_file)
    strategy, shifted_to_orig = choose_mapping_strategy(alpha0_dir, originals)
    if sum(1 for s in [p.stem for p in list_images(alpha0_dir)] if s in shifted_to_orig) == 0:
        print(f"[warn] No matches found for any strategy. 'vs original' bars may be empty.")

    # Results accumulators
    rows = []
    ssim_vs_a0: Dict[float, List[float]] = {a: [] for a in alphas if a != 0.0}
    ssim_vs_orig: Dict[float, List[float]] = {a: [] for a in alphas}  # include 0.0 too

    # Iterate
    for a in alphas:
        a_dir = alpha_dirs[a]
        for p_a in list_images(a_dir):
            stem = p_a.stem

            # vs alpha_0
            if a != 0.0:
                p0 = a_dir.parent / f"alpha_+0" / f"{stem}{p_a.suffix}"
                if not p0.exists():
                    # try also plain 'alpha_0' just in case
                    p0_alt = a_dir.parent / f"alpha_0" / f"{stem}{p_a.suffix}"
                    p0 = p0 if p0.exists() else p0_alt
                if p0.exists():
                    try:
                        val = compute_ssim_pair(p_a, p0)
                        ssim_vs_a0[a].append(val)
                        rows.append({
                            "subject": subject_str,
                            "model": model,
                            "comparison": "alpha_vs_alpha0",
                            "alpha": a,
                            "image_id": stem,
                            "path_alpha": str(p_a),
                            "path_ref": str(p0),
                            "ssim": val,
                        })
                    except Exception:
                        pass

            # vs original via chosen strategy
            p_orig = shifted_to_orig.get(stem)
            if p_orig and Path(p_orig).exists():
                try:
                    val = compute_ssim_pair(p_a, Path(p_orig))
                    ssim_vs_orig[a].append(val)
                    rows.append({
                        "subject": subject_str,
                        "model": model,
                        "comparison": "alpha_vs_original",
                        "alpha": a,
                        "image_id": stem,
                        "path_alpha": str(p_a),
                        "path_ref": str(p_orig),
                        "ssim": val,
                    })
                except Exception:
                    pass

    # Save CSV
    out_base = plots_dir / "ssim"
    ensure_dir(out_base)
    csv_path = out_base / f"{subject_str}_{model}_ssim_per_image.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject","model","comparison","alpha","image_id","path_alpha","path_ref","ssim"
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Helper: barplot
    def make_barplot(data_map: Dict[float, List[float]], title: str, png_path: Path):
        # keep only alphas with at least min_n items
        means, counts = {}, {}
        for k, vals in data_map.items():
            if len(vals) >= min_n:
                means[k] = float(np.mean(vals))
                counts[k] = len(vals)  # still computed, but not shown

        xs = sorted(means.keys())
        if not xs:
            print(f"[warn] No bars for plot: {png_path.name} (min_n={min_n})")

        ys = [means[k] for k in xs]

        plt.figure(figsize=(9, 5))
        labels = [f"{k:+g}" for k in xs]
        bars = plt.bar(labels, ys)

        # Annotate each bar with the mean SSIM value
        for b, y in zip(bars, ys):
            h = b.get_height()
            plt.text(
                b.get_x() + b.get_width()/2.0,
                h,
                f"{y:.3f}",            # <-- exact mean SSIM shown here
                ha="center",
                va="bottom",
                fontsize=10
            )

        plt.ylim(0, 1.0)
        plt.xlabel("alpha level (k)")
        plt.ylabel("Mean SSIM")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()


    # Plots
    out_vs_a0 = out_base / f"{subject_str}_{model}_mean_ssim_vs_alpha0.png"
    make_barplot(
        ssim_vs_a0,
        title=f"{subject_str} · {model}: mean SSIM (alpha_k vs alpha_0)",
        png_path=out_vs_a0,
    )

    out_vs_orig = out_base / f"{subject_str}_{model}_mean_ssim_vs_original.png"
    make_barplot(
        ssim_vs_orig,
        title=f"{subject_str} · {model}: mean SSIM (alpha_k vs original)  [{strategy}]",
        png_path=out_vs_orig,
    )

    # Summary JSON
    summary = {
        "subject": subject_str,
        "model": model,
        "match_strategy": strategy,
        "alphas_available": [float(a) for a in sorted(alpha_dirs.keys())],
        "alphas_used": [float(a) for a in alphas],
        "csv_path": str(csv_path),
        "plot_vs_alpha0": str(out_vs_a0),
        "plot_vs_original": str(out_vs_orig),
        "min_images_per_bar": int(min_n),
    }
    with (out_base / f"{subject_str}_{model}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] {subject_str} | {model} →")
    print(f"  CSV:  {csv_path}")
    print(f"  Plot: {out_vs_a0}")
    print(f"  Plot: {out_vs_orig}")
    return {"csv": str(csv_path), "plot_vs_alpha0": str(out_vs_a0), "plot_vs_original": str(out_vs_orig)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, required=True, help="Subject number, e.g., 1 for subj01")
    ap.add_argument("--shifted_base", type=str, default="/home/rothermm/THINGS/03_results/vdvae_shifted",
                    help="Base of shifted images (append subjXX internally).")
    ap.add_argument("--stim_paths_root", type=str, default="/home/rothermm/THINGS/02_data/preprocessed_data",
                    help="Root with subjXX/test_image_paths.txt")
    ap.add_argument("--plots_dir", type=str, default="/home/rothermm/THINGS/03_results/plots",
                    help="Where to save plots (creates 'ssim' subfolder).")
    ap.add_argument("--models", type=str, nargs="+", default=["emonet", "memnet"],
                    choices=["emonet", "memnet"])
    ap.add_argument("--alphas", type=float, nargs="*", default=None,
                    help="Optional explicit alpha list (e.g., -4 -3 -2 -1 -0.5 0 0.5 1 2 3 4).")
    ap.add_argument("--min_n", type=int, default=5, help="Min #images per alpha bar to include.")
    args = ap.parse_args()

    sub_pad = f"subj{args.sub:02d}"
    shifted_subject_base = Path(args.shifted_base) / sub_pad
    assert shifted_subject_base.is_dir(), f"Missing subject shifted dir: {shifted_subject_base}"

    stim_paths_file = Path(args.stim_paths_root) / sub_pad / "test_image_paths.txt"
    assert stim_paths_file.is_file(), f"Missing: {stim_paths_file}"

    plots_dir = Path(args.plots_dir).resolve()

    results = {}
    for m in args.models:
        results[m] = compute_for_model(
            subject_str=sub_pad,
            model=m,
            shifted_base=shifted_subject_base,
            stim_paths_file=stim_paths_file,
            plots_dir=plots_dir,
            selected_alphas=args.alphas,
            min_n=args.min_n,
        )
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
