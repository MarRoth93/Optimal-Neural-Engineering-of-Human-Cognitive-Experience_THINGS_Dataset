#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_split_800.py

Purpose
-------
Create a metadata file where exactly N unique images are assigned to TEST by
rewriting the 'trial_type' column based on image IDs. Row order is preserved.

Why this works with your pipeline
---------------------------------
- You keep using your existing `04_split_and_average_things.py` unchanged.
- TEST duplicates are still averaged there.
- Train remains per-trial by default (or averaged if you later pass --avg_train).
- Alignment between saved matrices and ID lists remains correct.

Usage
-----
python make_split_800.py \
  --stimulus_tsv /path/subj01_stimulus-metadata.tsv \
  --out_tsv      /path/subj01_split800.tsv \
  --id_col stimulus \
  --strategy first \
  --seed 42 \
  --n_test 800
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rewrite 'trial_type' to mark exactly N unique IDs as TEST.")
    ap.add_argument("--stimulus_tsv", type=Path, required=True,
                    help="Input stimulus metadata (TSV/CSV).")
    ap.add_argument("--out_tsv", type=Path, required=True,
                    help="Output TSV with updated 'trial_type'.")
    ap.add_argument("--id_col", type=str, default="stimulus",
                    help="Column identifying the image per trial (default: stimulus).")
    ap.add_argument("--strategy", choices=["first", "random"], default="first",
                    help="How to pick the TEST IDs: 'first' = first N by first appearance; 'random' = random N.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed used when strategy='random'.")
    ap.add_argument("--n_test", type=int, default=800,
                    help="Number of unique images to set as TEST.")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load; auto-detect CSV vs TSV
    df = pd.read_csv(args.stimulus_tsv, sep=None, engine="python")

    if args.id_col not in df.columns:
        raise KeyError(f"Column '{args.id_col}' not found in {args.stimulus_tsv}")

    # Normalize IDs to str for consistent matching
    ids = df[args.id_col].astype(str).to_numpy()

    # Unique IDs in *first-appearance* order (stable)
    # (pd.unique preserves order; np.unique would sort.)
    uniq_ids = pd.unique(ids)
    total_unique = len(uniq_ids)
    if args.n_test > total_unique:
        raise ValueError(f"Requested {args.n_test} test images, but only {total_unique} unique found.")

    # Choose which IDs become TEST
    if args.strategy == "first":
        chosen_test_ids = set(uniq_ids[:args.n_test])
    else:
        rng = np.random.default_rng(args.seed)
        chosen_test_ids = set(rng.choice(uniq_ids, size=args.n_test, replace=False))

    # Rewrite trial_type (row order preserved)
    df["trial_type"] = np.where(df[args.id_col].astype(str).isin(chosen_test_ids), "test", "train")

    # Write output (TSV)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)

    # Small summary (also useful sanity checks)
    n_rows = len(df)
    n_test_rows = int((df["trial_type"] == "test").sum())
    # Number of unique TEST images should equal n_test
    test_unique = df.loc[df["trial_type"] == "test", args.id_col].astype(str).nunique()

    print("---- make_split_800 summary ----")
    print(f"in:   {args.stimulus_tsv}")
    print(f"out:  {args.out_tsv}")
    print(f"id_col: {args.id_col}")
    print(f"strategy: {args.strategy} | seed: {args.seed}")
    print(f"rows: {n_rows} | unique IDs total: {total_unique}")
    print(f"unique TEST IDs: {test_unique} (requested {args.n_test})")
    print(f"TEST rows (with repeats): {n_test_rows}")
    if test_unique != args.n_test:
        raise RuntimeError(
            f"Post-check failed: unique TEST IDs = {test_unique}, expected {args.n_test}."
        )
    print("OK.")


if __name__ == "__main__":
    main()
