#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_split_800.py

Goal
----
Guarantee that all ORIGINAL test images (e.g., the ~100 shown multiple times)
remain in TEST, and then add extra images until exactly N unique TEST IDs.

How
---
1) Read the ORIGINAL metadata (with its current 'trial_type').
2) Pin all unique IDs currently labeled as test.
3) From the remaining unique IDs (first-appearance order), choose extras by
   strategy ('first' or 'random') to reach N unique TEST IDs total.
4) Rewrite 'trial_type' accordingly (row order preserved) and save a new TSV.

Usage
-----
python make_split_800.py \
  --stimulus_tsv /path/sub-01_task-things_stimulus-metadata.tsv \
  --out_tsv      /path/sub-01_task-things_stimulus-metadata_split800.tsv \
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
    ap = argparse.ArgumentParser(
        description="Rewrite 'trial_type' to mark exactly N unique IDs as TEST, "
                    "while GUARANTEEING all original test IDs stay in TEST."
    )
    ap.add_argument("--stimulus_tsv", type=Path, required=True,
                    help="Input stimulus metadata (TSV/CSV) with existing 'trial_type'.")
    ap.add_argument("--out_tsv", type=Path, required=True,
                    help="Output TSV with updated 'trial_type'.")
    ap.add_argument("--id_col", type=str, default="stimulus",
                    help="Column identifying the image per trial (default: stimulus).")
    ap.add_argument("--strategy", choices=["first", "random"], default="first",
                    help="How to pick ADDITIONAL TEST IDs: 'first' = first-appearance order; "
                         "'random' = RNG sample from remaining.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed used when strategy='random'.")
    ap.add_argument("--n_test", type=int, default=800,
                    help="Total number of unique TEST images desired (>= original test size).")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load; auto-detect CSV vs TSV
    df = pd.read_csv(args.stimulus_tsv, sep=None, engine="python")

    # Basic checks
    if "trial_type" not in df.columns:
        raise KeyError(f"'trial_type' column not found in {args.stimulus_tsv}")
    if args.id_col not in df.columns:
        raise KeyError(f"ID column '{args.id_col}' not found in {args.stimulus_tsv}")

    # Normalize
    id_series = df[args.id_col].astype(str)
    trial_series = df["trial_type"].astype(str).str.lower()

    # Unique IDs in first-appearance order (stable)
    uniq_ids_all = pd.unique(id_series.to_numpy())
    total_unique = len(uniq_ids_all)

    # PIN: all original TEST unique IDs
    pinned_test_ids = pd.unique(id_series[trial_series == "test"])
    pinned_set = set(map(str, pinned_test_ids))
    n_pinned = len(pinned_set)

    if n_pinned == 0:
        raise ValueError("No original TEST images found in the input metadata. "
                         "This script expects an existing test set to pin.")
    if args.n_test < n_pinned:
        raise ValueError(f"Requested n_test ({args.n_test}) is smaller than the original "
                         f"TEST set size ({n_pinned}). Increase --n_test to at least {n_pinned}.")

    # Build candidate pool (respect first-appearance order, exclude pinned)
    candidates = [u for u in uniq_ids_all if u not in pinned_set]
    need = args.n_test - n_pinned

    if need > len(candidates):
        raise ValueError(f"Need {need} additional TEST IDs, but only {len(candidates)} candidates remain "
                         f"(total unique={total_unique}, pinned={n_pinned}). Reduce --n_test.")

    # Choose extras
    if need > 0:
        if args.strategy == "first":
            extras = candidates[:need]
        else:
            rng = np.random.default_rng(args.seed)
            extras = list(rng.choice(candidates, size=need, replace=False))
    else:
        extras = []

    chosen_test_ids = pinned_set.union(map(str, extras))

    # Rewrite trial_type (row order preserved)
    df["trial_type"] = np.where(id_series.isin(chosen_test_ids), "test", "train")

    # Save
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)

    # Summaries
    new_test_unique = df.loc[df["trial_type"] == "test", args.id_col].astype(str).nunique()
    n_rows = len(df)
    n_test_rows = int((df["trial_type"] == "test").sum())
    kept_pinned = len({*pinned_set}.intersection(set(df.loc[df["trial_type"] == "test", args.id_col].astype(str))))

    print("---- make_split_800 summary ----")
    print(f"in:   {args.stimulus_tsv}")
    print(f"out:  {args.out_tsv}")
    print(f"id_col: {args.id_col}")
    print(f"strategy: {args.strategy} | seed: {args.seed}")
    print(f"rows: {n_rows} | unique IDs total: {total_unique}")
    print(f"original TEST unique: {n_pinned}  | extras added: {len(extras)}  | target n_test: {args.n_test}")
    print(f"unique TEST IDs (new): {new_test_unique} (expected {args.n_test})")
    print(f"pinned kept in TEST: {kept_pinned} (expected {n_pinned})")
    print(f"TEST rows (with repeats): {n_test_rows}")
    if new_test_unique != args.n_test or kept_pinned != n_pinned:
        raise RuntimeError("Post-check failed: counts do not match expectations.")
    print("OK.")


if __name__ == "__main__":
    main()
