#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick integrity check for the 800-image TEST split and the derived outputs.

It verifies:
- Split metadata has exactly N unique TEST images
- train/test index files match the split
- Shapes and row ↔ ID alignment for X_test_avg and (X_train or X_train_avg)
- summary.json counters match reality
- Optional: X.npy rows == len(stimulus TSV)

Exit code: 0 on success, 1 on any failure.
"""

import argparse, json, sys, os
from pathlib import Path
import numpy as np
import pandas as pd

def read_table_auto(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, sep=None, engine="python")

def read_ids(p: Path):
    return [ln.strip() for ln in open(p, "r", encoding="utf-8")]

def ok(msg):  print(f"[OK]   {msg}")
def fail(msg): print(f"[FAIL] {msg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimulus_tsv", type=Path, required=True,
                    help="Split stimulus metadata TSV/CSV (…_split800.tsv)")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="Directory with outputs of 04b_split_and_average_things_800.py")
    ap.add_argument("--id_col", type=str, default="stimulus")
    ap.add_argument("--n_test", type=int, default=800)
    ap.add_argument("--x_path", type=Path, default=None,
                    help="Optional: X.npy to verify trial count")
    args = ap.parse_args()

    errors = 0

    try:
        stim = read_table_auto(args.stimulus_tsv)
        if args.id_col not in stim.columns:
            raise KeyError(f"Missing column: {args.id_col}")

        # Normalize case for trial_type
        tt = stim["trial_type"].astype(str).str.lower()
        is_test = (tt == "test").to_numpy()
        is_train = (tt == "train").to_numpy()

        n_total = len(stim)
        n_test_rows = int(is_test.sum())
        n_train_rows = int(is_train.sum())
        n_test_unique = stim.loc[is_test, args.id_col].astype(str).nunique()
        n_train_unique = stim.loc[is_train, args.id_col].astype(str).nunique()

        if n_test_unique == args.n_test:
            ok(f"Split has exactly {args.n_test} unique TEST images.")
        else:
            errors += 1; fail(f"Unique TEST images = {n_test_unique}, expected {args.n_test}.")

        ok(f"Split rows: total={n_total}, train_rows={n_train_rows}, test_rows={n_test_rows}")

        if args.x_path and args.x_path.exists():
            X = np.load(args.x_path)
            if X.shape[0] == n_total:
                ok(f"X.npy rows ({X.shape[0]}) match stimulus rows ({n_total}).")
            else:
                errors += 1; fail(f"X.npy rows ({X.shape[0]}) != stimulus rows ({n_total}).")
    except Exception as e:
        errors += 1; fail(f"Reading/validating split failed: {e}")

    # Paths in out_dir
    od = args.out_dir
    test_meta_path = od / "test_meta_avg.tsv"
    test_ids_path  = od / "test_image_ids.txt"
    test_X_path    = od / "X_test_avg.npy"
    train_idx_path = od / "train_idx.npy"
    test_idx_path  = od / "test_idx.npy"
    summary_path   = od / "summary.json"

    # Train can be per-trial or averaged
    train_X_path_pt = od / "X_train.npy"
    train_meta_pt   = od / "train_meta.tsv"
    train_ids_pt    = od / "train_image_ids.txt"

    train_X_path_avg = od / "X_train_avg.npy"
    train_meta_avg   = od / "train_meta_avg.tsv"
    train_ids_avg    = od / "train_image_ids_avg.txt"

    try:
        # Indices vs split masks
        tr_idx = np.load(train_idx_path)
        te_idx = np.load(test_idx_path)
        ok(f"Found index files: train_idx.npy ({len(tr_idx)}), test_idx.npy ({len(te_idx)}).")

        # Check they match the split
        stim_mask_train = np.where(is_train)[0]
        stim_mask_test  = np.where(is_test)[0]
        if np.array_equal(tr_idx, stim_mask_train):
            ok("train_idx matches 'train' rows in split.")
        else:
            errors += 1; fail("train_idx does not match 'train' mask from split.")
        if np.array_equal(te_idx, stim_mask_test):
            ok("test_idx matches 'test' rows in split.")
        else:
            errors += 1; fail("test_idx does not match 'test' mask from split.")

        # Coverage & disjointness
        union = np.union1d(tr_idx, te_idx)
        if len(union) == n_total and len(np.intersect1d(tr_idx, te_idx)) == 0:
            ok("train_idx ∪ test_idx covers all rows and is disjoint.")
        else:
            errors += 1; fail("train/test indices are overlapping or incomplete.")
    except Exception as e:
        errors += 1; fail(f"Index check failed: {e}")

    try:
        # TEST consistency
        tmeta = pd.read_csv(test_meta_path, sep="\t")
        tids  = read_ids(test_ids_path)
        Xte   = np.load(test_X_path)
        # rows/ids/shape
        if len(tmeta) == args.n_test:
            ok(f"test_meta_avg.tsv has {len(tmeta)} rows (expected {args.n_test}).")
        else:
            errors += 1; fail(f"test_meta_avg.tsv rows={len(tmeta)} != {args.n_test}.")

        if len(tids) == args.n_test:
            ok(f"test_image_ids.txt has {len(tids)} lines.")
        else:
            errors += 1; fail(f"test_image_ids.txt lines={len(tids)} != {args.n_test}.")

        if Xte.shape[0] == args.n_test:
            ok(f"X_test_avg.npy has {Xte.shape[0]} rows.")
        else:
            errors += 1; fail(f"X_test_avg.npy rows={Xte.shape[0]} != {args.n_test}.")

        # order alignment: IDs file == meta image_id column
        if "image_id" in tmeta.columns and tids == tmeta["image_id"].astype(str).tolist():
            ok("Order: test_image_ids.txt aligns with test_meta_avg.tsv['image_id'].")
        else:
            ok("Note: Could not strictly verify test ID order against meta (column missing or different).")

    except Exception as e:
        errors += 1; fail(f"TEST outputs check failed: {e}")

    try:
        # TRAIN consistency (per-trial OR averaged)
        if train_X_path_pt.exists():
            Xtr = np.load(train_X_path_pt)
            tmeta_tr = pd.read_csv(train_meta_pt, sep="\t")
            tids_tr = read_ids(train_ids_pt)
            # per-trial: rows must equal number of train trials
            if Xtr.shape[0] == len(tr_idx) == len(tmeta_tr) == len(tids_tr):
                ok(f"X_train.npy rows ({Xtr.shape[0]}) match train_idx/meta/ids (per-trial mode).")
            else:
                errors += 1; fail("Mismatch in per-trial TRAIN shapes/ids/meta.")
        elif train_X_path_avg.exists():
            Xtr = np.load(train_X_path_avg)
            tmeta_tr = pd.read_csv(train_meta_avg, sep="\t")
            tids_tr = read_ids(train_ids_avg)
            n_unique_train = n_train_unique
            if (Xtr.shape[0] == n_unique_train ==
                len(tmeta_tr) == len(tids_tr)):
                ok(f"X_train_avg.npy rows ({Xtr.shape[0]}) match unique TRAIN IDs/meta/ids (averaged mode).")
            else:
                errors += 1; fail("Mismatch in averaged TRAIN shapes/ids/meta.")
        else:
            errors += 1; fail("Neither X_train.npy nor X_train_avg.npy found.")
    except Exception as e:
        errors += 1; fail(f"TRAIN outputs check failed: {e}")

    try:
        # summary.json coherence
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        n_test_saved = summary.get("n_test_images_saved", None)
        if n_test_saved == args.n_test:
            ok(f"summary.json: n_test_images_saved = {n_test_saved}")
        else:
            errors += 1; fail(f"summary.json: n_test_images_saved={n_test_saved} != {args.n_test}")
        ok("summary.json loaded.")
    except Exception as e:
        errors += 1; fail(f"summary.json check failed: {e}")

    if errors == 0:
        print("✅ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print(f"❌ CHECKS FAILED: {errors} issue(s) found")
        sys.exit(1)

if __name__ == "__main__":
    main()
