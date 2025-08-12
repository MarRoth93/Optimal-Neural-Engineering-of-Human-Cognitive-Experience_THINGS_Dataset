#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_extract_betas_to_numpy.py
- Loads a binary VC mask (T1w space) for sub-01
- Applies it to all single-trial NIfTI betas (you provide a glob)
- Stacks masked vectors into X.npy: shape (n_trials, n_vox_in_mask)
- Writes meta.tsv (one row per trial) and basic validation

Tip: Start with a small --limit to dry-run the pipeline.
"""

from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from tqdm import tqdm
from glob import glob

def eprint(*a, **k): print(*a, file=sys.stderr, **k)
def require(cond, msg):
    if not cond:
        eprint("[ERROR]", msg); sys.exit(1)

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

def collect_events_counts(events_root: Path) -> int:
    """Count total rows across all *_events.tsv under sub-01/ses-things*."""
    tsvs = sorted((events_root).rglob("*_events.tsv"))
    n = 0
    for p in tsvs:
        # only task-things sessions
        if "ses-things" not in p.as_posix(): 
            continue
        try:
            n += sum(1 for _ in open(p, "r", encoding="utf-8")) - 1  # minus header
        except Exception:
            pass
    return n

def parse_session_run_from_name(name: str):
    # capture ses-XX and run-XX if present
    ses = None; run = None
    m = re.search(r"ses-([0-9A-Za-z]+)", name)
    if m: ses = m.group(1)
    m = re.search(r"run-([0-9A-Za-z]+)", name)
    if m: run = m.group(1)
    return ses, run

def main():
    ap = argparse.ArgumentParser(description="Extract masked NIfTI betas to X.npy for sub-01")
    ap.add_argument("--mask", type=Path, required=True, help="binary VC mask NIfTI (T1w space)")
    ap.add_argument("--betas_glob", type=str, required=True,
                    help="glob for single-trial beta NIfTIs (e.g., '/.../ICA-betas/sub-01/nifti/*beta*.nii.gz')")
    ap.add_argument("--out_dir", type=Path, required=True, help="output dir for X.npy + meta.tsv")
    ap.add_argument("--events_root", type=Path, default=None, help="root with sub-01/ses-*/func/*_events.tsv for sanity check")
    ap.add_argument("--limit", type=int, default=None, help="process only first N files (dry-run/testing)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load mask
    eprint("[1/5] Loading mask:", args.mask)
    mask_img = nib.load(str(args.mask))
    mask = mask_img.get_fdata().astype(bool)
    n_vox = int(mask.sum())
    require(n_vox > 0, "Mask has 0 voxels set.")
    eprint(f"[INFO] mask voxels: {n_vox}")

    # 2) Collect beta files
    files = sorted(glob(args.betas_glob), key=natural_key)
    require(len(files) > 0, f"No NIfTI files matched glob: {args.betas_glob}")
    if args.limit:
        files = files[:args.limit]
        eprint(f"[INFO] limit active: using first {len(files)} files")

    # 3) Preflight: ensure shapes match mask; pick a reference image
    ref_img = nib.load(files[0])
    # If mask is not in exact same grid, resample once:
    if not np.allclose(mask_img.affine, ref_img.affine) or mask_img.shape != ref_img.shape[:3]:
        eprint("[INFO] Resampling mask to beta grid (nearest neighbor) …")
        mask_img_rs = image.resample_to_img(mask_img, ref_img, interpolation="nearest")
        mask = mask_img_rs.get_fdata().astype(bool)
        n_vox = int(mask.sum())
        require(n_vox > 0, "Resampled mask has 0 voxels set.")
    eprint(f"[INFO] reference beta shape: {ref_img.shape} | mask sum: {n_vox}")

    # 4) Create memory-mapped X.npy (trials x vox)
    X_path = args.out_dir / "X.npy"
    n_trials = len(files)
    eprint(f"[2/5] Allocating memmap for X: shape=({n_trials}, {n_vox})")
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(n_trials, n_vox))

    # 5) Extract masked vectors
    meta_rows = []
    for idx, fpath in enumerate(tqdm(files, desc="[3/5] Extracting")):
        img = nib.load(fpath)
        dat = img.get_fdata()
        if dat.ndim == 4:
            if dat.shape[3] != 1:
                require(False, f"Found 4D NIfTI with >1 volumes: {fpath}")
            dat = dat[..., 0]
        require(dat.shape == mask.shape, f"Shape mismatch for {fpath}: {dat.shape} vs {mask.shape}")
        X[idx, :] = dat[mask].astype(np.float32)

        name = Path(fpath).name
        ses, run = parse_session_run_from_name(name)
        meta_rows.append({"index": idx, "file": name, "session": ses, "run": run})

    # flush memmap to disk
    del X

    # 6) Write meta.tsv
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(args.out_dir / "meta.tsv", sep="\t", index=False)
    eprint(f"[4/5] Saved meta.tsv with {len(meta_df)} rows")

    # 7) Optional events sanity check
    if args.events_root is not None:
        expected = collect_events_counts(args.events_root / "sub-01")
        eprint(f"[5/5] events.tsv total rows (task-things): {expected}")
        if expected != n_trials:
            eprint(f"[WARN] #betas({n_trials}) != #events rows({expected}). Check your glob or sessions subset.")
        else:
            eprint("[OK] beta files count matches events rows.")

    eprint("[DONE] X.npy written to", X_path.resolve())

if __name__ == "__main__":
    main()
