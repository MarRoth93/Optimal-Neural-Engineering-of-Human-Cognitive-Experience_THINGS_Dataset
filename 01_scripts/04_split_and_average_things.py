#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_split_and_average_things.py

Inputs:
- X.npy  (from 03_extract_h5_to_numpy_v2.py)  shape: (n_trials, n_vox)
- stimulus metadata TSV for the SAME subject, containing at least:
  - 'trial_type'  (values like 'train' / 'test')
  - an ID column identifying the stimulus per trial (default: 'stimulus')

What it does:
- Loads X and the stimulus metadata (auto-detects CSV vs TSV delimiter)
- Splits into train and test by 'trial_type'
- TEST: groups by ID and averages repeated trials → X_test_avg.npy
- TRAIN: by default keeps per-trial rows → X_train.npy
  (option --avg_train if you want to average by ID in train as well)
- Saves ID lists in the SAME order as the saved matrices, plus CSV metadata.

Outputs (in --out_dir):
- X_train.npy / train_meta.tsv / train_image_ids.txt
- X_test_avg.npy / test_meta_avg.tsv / test_image_ids.txt
- train_idx.npy / test_idx.npy  (trial indices from the original X order)
- summary.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def read_table_auto(path: Path) -> pd.DataFrame:
    # Handle both comma- and tab-separated files gracefully
    return pd.read_csv(path, sep=None, engine="python")

def avg_by_id(X_rows: np.ndarray, ids: np.ndarray):
    """Average X_rows (n_trials_subset, n_vox) per unique id in 'ids'."""
    df = pd.DataFrame({"sid": ids})
    groups = df.groupby("sid").indices
    uniq = []
    rows = []
    for sid, idxs in groups.items():
        uniq.append(sid)
        rows.append(X_rows[list(idxs)].mean(axis=0))
    return np.vstack(rows), uniq, groups

def main():
    ap = argparse.ArgumentParser(description="Split THINGS into train/test and average repeats in test.")
    ap.add_argument("--x_path", type=Path, required=True, help="Path to X.npy (trials x voxels)")
    ap.add_argument("--stimulus_tsv", type=Path, required=True, help="Path to stimulus metadata TSV/CSV")
    ap.add_argument("--out_dir", type=Path, required=True, help="Output directory")
    ap.add_argument("--id_col", type=str, default="stimulus", help="Column to identify images (default: 'stimulus')")
    ap.add_argument("--avg_train", action="store_true", help="Also average repeats in TRAIN (default: keep per-trial)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load inputs
    X = np.load(args.x_path)  # (n_trials, n_vox)
    stim = read_table_auto(args.stimulus_tsv)

    # Basic checks
    assert "trial_type" in stim.columns, "trial_type column not found in stimulus metadata"
    assert args.id_col in stim.columns, f"{args.id_col} not found in stimulus metadata"
    assert len(stim) == X.shape[0], f"Trial count mismatch: X has {X.shape[0]} rows, stim has {len(stim)} rows"

    # Normalize ID column to string
    stim[args.id_col] = stim[args.id_col].astype(str)

    # 2) Build boolean masks and index arrays
    is_train = (stim["trial_type"].astype(str).str.lower() == "train").to_numpy()
    is_test  = (stim["trial_type"].astype(str).str.lower() == "test").to_numpy()

    train_idx = np.where(is_train)[0]
    test_idx  = np.where(is_test)[0]

    # Save trial indices for traceability
    np.save(args.out_dir / "train_idx.npy", train_idx)
    np.save(args.out_dir / "test_idx.npy", test_idx)

    # 3) TRAIN: per-trial by default OR average by id
    train_ids = stim.loc[is_train, args.id_col].to_numpy()
    X_train_rows = X[train_idx]

    if args.avg_train:
        X_train, train_id_list, train_groups = avg_by_id(X_train_rows, train_ids)
        # Build a compact meta with the averaged groups
        train_meta_avg = pd.DataFrame({
            "image_id": train_id_list,
            "n_trials_averaged": [len(train_groups[sid]) for sid in train_id_list],
        })
        X_train_out = X_train.astype(np.float32)
        train_meta_out = train_meta_avg
        train_id_seq = train_id_list
        train_matrix_name = "X_train_avg.npy"
        train_meta_name = "train_meta_avg.tsv"
        train_ids_name = "train_image_ids_avg.txt"
    else:
        # Keep per-trial; just pass through
        X_train_out = X_train_rows.astype(np.float32)
        train_meta_out = stim.loc[is_train].reset_index(drop=True)
        train_id_seq = train_ids.tolist()
        train_matrix_name = "X_train.npy"
        train_meta_name = "train_meta.tsv"
        train_ids_name = "train_image_ids.txt"

    # 4) TEST: average by id (this is the standard)
    test_ids = stim.loc[is_test, args.id_col].to_numpy()
    X_test_avg, test_id_list, test_groups = avg_by_id(X[test_idx], test_ids)
    test_meta_avg = pd.DataFrame({
        "image_id": test_id_list,
        "n_trials_averaged": [len(test_groups[sid]) for sid in test_id_list],
    })
    X_test_out = X_test_avg.astype(np.float32)

    # 5) Save outputs
    np.save(args.out_dir / train_matrix_name, X_train_out)
    np.save(args.out_dir / "X_test_avg.npy", X_test_out)

    train_meta_out.to_csv(args.out_dir / train_meta_name, sep="\t", index=False)
    test_meta_avg.to_csv(args.out_dir / "test_meta_avg.tsv", sep="\t", index=False)

    # ID lists (order matches the saved matrices)
    with open(args.out_dir / train_ids_name, "w") as f:
        for sid in train_id_seq:
            f.write(f"{sid}\n")
    with open(args.out_dir / "test_image_ids.txt", "w") as f:
        for sid in test_id_list:
            f.write(f"{sid}\n")

    # Summary
    summary = {
        "x_path": str(args.x_path),
        "stimulus_tsv": str(args.stimulus_tsv),
        "out_dir": str(args.out_dir),
        "id_col": args.id_col,
        "avg_train": bool(args.avg_train),
        "X_shape": list(X.shape),
        "n_train_trials": int(is_train.sum()),
        "n_test_trials": int(is_test.sum()),
        "train_matrix": train_matrix_name,
        "test_matrix": "X_test_avg.npy",
        "n_train_rows_saved": int(X_train_out.shape[0]),
        "n_test_images_saved": int(X_test_out.shape[0]),
        "n_vox": int(X.shape[1]),
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Wrote:",
          args.out_dir / train_matrix_name,
          args.out_dir / "X_test_avg.npy",
          args.out_dir / train_meta_name,
          args.out_dir / "test_meta_avg.tsv")
    print("[INFO] Train rows saved:", X_train_out.shape, "| Test avg rows saved:", X_test_out.shape)

if __name__ == "__main__":
    main()
