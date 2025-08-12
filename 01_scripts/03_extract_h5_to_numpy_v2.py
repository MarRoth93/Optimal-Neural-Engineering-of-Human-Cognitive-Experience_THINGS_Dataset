#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_extract_h5_to_numpy_v2.py
HDF5 (pandas-style) -> X.npy for sub-01
Loads only selected VC voxels using axis1 (voxel_id) to avoid huge RAM.

Outputs:
- X.npy           (n_trials, n_vc_vox)  float32
- mask_idx.npy    (n_vc_vox,) int64     positions in HDF5 row axis
- vc_voxel_ids.npy (n_vc_vox,) int64    the voxel_id values selected (for traceability)
- meta.json       shapes, dataset info
- meta.tsv        trial_index rows
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

def eprint(*a, **k): print(*a, file=sys.stderr, **k)
def require(cond, msg):
    if not cond:
        eprint("[ERROR]", msg); sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, required=True,
                    help=".../sub-01_task-things_voxel-wise-responses.h5")
    ap.add_argument("--voxel_tsv", type=Path, required=True,
                    help=".../sub-01_task-things_voxel-metadata.tsv")
    ap.add_argument("--stimulus_tsv", type=Path, default=None,
                    help="optional: stimulus metadata for sanity (rows = trials)")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--zscore_train_frac", type=float, default=None,
                    help="optional: fit z-score on first frac of trials (e.g., 0.8)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load voxel metadata and select VC voxels
    df = pd.read_csv(args.voxel_tsv)
    require("voxel_id" in df.columns, "voxel_id column missing in voxel TSV")

    vc_cols = [
        "V1","V2","V3","hV4","VO1","VO2",
        "LO1 (prf)","LO2 (prf)","TO1","TO2","V3b","V3a",
        "lEBA","rEBA","lFFA","rFFA","lOFA","rOFA",
        "lSTS","rSTS","lPPA","rPPA","lRSC","rRSC",
        "lTOS","rTOS","lLOC","rLOC","IT"
    ]
    missing = [c for c in vc_cols if c not in df.columns]
    require(not missing, f"Missing ROI columns in voxel TSV: {missing}")

    vc_mask = (df[vc_cols].sum(axis=1) > 0).values
    vc_voxel_ids = df.loc[vc_mask, "voxel_id"].astype(np.int64).to_numpy()
    require(vc_voxel_ids.size > 0, "No VC voxels selected.")

    # 2) Open HDF5 and find arrays
    with h5py.File(args.h5, "r") as f:
        g = f["ResponseData"]
        axis1 = np.asarray(g["axis1"])              # voxel IDs, length n_vox_total
        data  = g["block0_values"]                  # (voxels, trials), float32

        # Map VC voxel_ids -> row positions in HDF5
        # Build dict from voxel_id -> row index
        id_to_pos = {int(v): i for i, v in enumerate(axis1)}
        missing_ids = [int(v) for v in vc_voxel_ids if int(v) not in id_to_pos]
        require(len(missing_ids) == 0,
                f"{len(missing_ids)} VC voxel_ids not found in HDF5 axis1 (e.g., {missing_ids[:5]})")

        row_pos = np.array([id_to_pos[int(v)] for v in vc_voxel_ids], dtype=np.int64)
        # Slice rows: (selected_voxels, trials)
        # h5py supports fancy indexing, but may load in chunks; that’s fine.
        X_sel = data[row_pos, :]    # shape: (n_vc_vox, n_trials)

    # 3) Transpose to (trials, voxels)
    X = np.asarray(X_sel).T        # (n_trials, n_vc_vox)

    # 4) Optional z-score (train-only by fraction)
    zstats = None
    if args.zscore_train_frac and 0 < args.zscore_train_frac < 1.0:
        n_trials = X.shape[0]
        n_train  = max(1, int(round(args.zscore_train_frac * n_trials)))
        mu = X[:n_train].mean(axis=0)
        sd = X[:n_train].std(axis=0, ddof=0); sd[sd == 0] = 1.0
        X = (X - mu) / sd
        zstats = {"n_train": n_train, "mu_len": len(mu), "sd_len": len(sd)}

    # 5) Optional sanity: trials expected
    trials_expected = None
    if args.stimulus_tsv and Path(args.stimulus_tsv).exists():
        stim = pd.read_csv(args.stimulus_tsv, sep="\t")
        trials_expected = int(len(stim))
        if trials_expected != X.shape[0]:
            eprint(f"[WARN] HDF5 trials ({X.shape[0]}) != stimulus TSV rows ({trials_expected})")

    # 6) Save outputs
    np.save(args.out_dir / "X.npy", X.astype(np.float32))
    np.save(args.out_dir / "mask_idx.npy", row_pos)            # positions in HDF5 row axis
    np.save(args.out_dir / "vc_voxel_ids.npy", vc_voxel_ids)   # the actual voxel_id values

    meta = {
        "subject": "sub-01",
        "h5_file": str(args.h5),
        "dataset": "ResponseData/block0_values",
        "h5_shape": [int(211339), int(9840)],   # from your printout
        "X_shape": list(X.shape),
        "n_vc_vox": int(X.shape[1]),
        "n_trials": int(X.shape[0]),
        "zscore": zstats,
        "trials_expected": trials_expected
    }
    with open(args.out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # simple trial meta
    pd.DataFrame({"trial_index": np.arange(X.shape[0])}).to_csv(args.out_dir / "meta.tsv", sep="\t", index=False)

    eprint("[DONE] Saved:",
           args.out_dir / "X.npy",
           args.out_dir / "mask_idx.npy",
           args.out_dir / "vc_voxel_ids.npy",
           args.out_dir / "meta.json",
           args.out_dir / "meta.tsv")

if __name__ == "__main__":
    main()
