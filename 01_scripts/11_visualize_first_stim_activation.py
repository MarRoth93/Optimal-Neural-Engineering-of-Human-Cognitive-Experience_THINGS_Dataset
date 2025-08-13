#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_visualize_first_stim_activation.py

Visualize (a) the voxel-wise activation for the FIRST stimulus and
(b) the voxel mask used (from the HDF5 selection).

It will:
- Load fMRI data (defaults to X_test_avg.npy if present, else X.npy) and take the first row
- Map voxel values to 3D using voxel coordinates from the voxel-metadata TSV
- Save beautiful overlays (stat map + glass brain) and a histogram of activations
- Save a NIfTI volume for the 'used voxels' mask and for the activation map

Defaults are for subj01; adjust with CLI args if needed.

Usage:
python 11_visualize_first_stim_activation.py --sub 1
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt


def load_activation_vector(preproc_dir: Path):
    """
    Prefer X_test_avg.npy if present (matches your test split),
    otherwise fall back to X.npy (all trials in original order).
    Returns: (vector shape (n_vox,), label_str)
    """
    x_test = preproc_dir / "X_test_avg.npy"
    x_all  = preproc_dir / "X.npy"
    if x_test.exists():
        X = np.load(x_test)  # (n_test_images, n_vox)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError(f"Unexpected shape for {x_test}: {X.shape}")
        return X[0].astype(np.float32), "X_test_avg[0]"
    elif x_all.exists():
        X = np.load(x_all)  # (n_trials, n_vox)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError(f"Unexpected shape for {x_all}: {X.shape}")
        return X[0].astype(np.float32), "X[0]"
    else:
        raise FileNotFoundError(f"Neither {x_test} nor {x_all} found.")


def main():
    ap = argparse.ArgumentParser(description="Visualize first-stimulus activation and used voxel mask.")
    ap.add_argument("--sub", type=int, default=1, help="Subject number (e.g., 1)")
    ap.add_argument("--brainmask", type=Path, default=Path("/home/rothermm/THINGS/02_data/derivatives/fmriprep/sub-01/anat/sub-01_space-T1w_mask.nii.gz"),
                    help="T1w-space brain mask NIfTI (used also as bg if no T1w provided)")
    ap.add_argument("--voxel_tsv", type=Path, default=Path("/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_voxel-metadata.tsv"),
                    help="Voxel metadata TSV with voxel_id, voxel_x/y/z, ROI columns")
    ap.add_argument("--preproc_dir", type=Path, default=None,
                    help="Directory containing X.npy / X_test_avg.npy / vc_voxel_ids.npy (default subjXX preproc dir)")
    ap.add_argument("--t1w_bg", type=Path, default=None,
                    help="Optional T1w image to use as background; falls back to brainmask")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output dir for figures and NIfTIs (default subjXX visualizations dir)")
    args = ap.parse_args()

    sp = f"{args.sub:02d}"
    if args.preproc_dir is None:
        args.preproc_dir = Path(f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sp}")
    if args.out_dir is None:
        args.out_dir = Path(f"/home/rothermm/THINGS/03_results/visualizations/subj{sp}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load background (mask and optional T1w) ---
    bm_img = nib.load(str(args.brainmask))
    bm = bm_img.get_fdata().astype(bool)
    shape, aff, hdr = bm.shape, bm_img.affine, bm_img.header
    bg_img = nib.load(str(args.t1w_bg)) if args.t1w_bg and Path(args.t1w_bg).exists() else bm_img

    # --- Load voxel metadata ---
    df = pd.read_csv(args.voxel_tsv, sep=",")
    required_cols = {"voxel_id", "voxel_x", "voxel_y", "voxel_z"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in voxel TSV: {missing}")

    # --- Load the list of used voxel IDs (from your HDF5 selection step) ---
    used_ids_path = args.preproc_dir / "vc_voxel_ids.npy"
    if not used_ids_path.exists():
        raise FileNotFoundError(f"Missing {used_ids_path}")
    used_ids = np.load(used_ids_path).astype(np.int64)

    # Align to metadata
    df_idx = df.set_index("voxel_id")
    missing_ids = [int(v) for v in used_ids if int(v) not in df_idx.index]
    if len(missing_ids) > 0:
        raise ValueError(f"{len(missing_ids)} used voxel_ids not found in voxel TSV (e.g., {missing_ids[:5]})")

    used_df = df_idx.loc[used_ids]
    I = used_df["voxel_x"].astype(int).to_numpy()
    J = used_df["voxel_y"].astype(int).to_numpy()
    K = used_df["voxel_z"].astype(int).to_numpy()

    # Bounds check & mask constrain
    valid = (I >= 0) & (I < shape[0]) & (J >= 0) & (J < shape[1]) & (K >= 0) & (K < shape[2])
    I, J, K = I[valid], J[valid], K[valid]

    # --- Build 'used voxel' binary mask volume ---
    used_mask_vol = np.zeros(shape, dtype=np.uint8)
    used_mask_vol[I, J, K] = 1
    used_mask_vol *= bm.astype(np.uint8)  # keep only inside brainmask
    used_mask_img = nib.Nifti1Image(used_mask_vol, aff, hdr)
    used_mask_nii = args.out_dir / "used_voxels_mask.nii.gz"
    nib.save(used_mask_img, str(used_mask_nii))

    # --- Load activation vector for the first stimulus ---
    vec, source_tag = load_activation_vector(args.preproc_dir)  # shape (n_used_vox,)
    if vec.shape[0] != len(I):
        # If some used voxels were invalid/out-of-bounds and got dropped, align vector
        # Build a boolean mask (all valid indices True)
        # Here we assume vec is in the SAME order as used_ids (that’s how you built X)
        keep_mask = valid  # same order as used_ids and thus as vec
        vec = vec[keep_mask]

    # --- Build activation 3D volume ---
    act_vol = np.zeros(shape, dtype=np.float32)
    act_vol[I, J, K] = vec
    act_vol *= bm.astype(np.float32)
    act_img = nib.Nifti1Image(act_vol, aff, hdr)
    act_nii = args.out_dir / "first_stim_activation.nii.gz"
    nib.save(act_img, str(act_nii))

    # --- Pretty plotting ---
    # Robust color scaling: symmetric around 0 with 98th percentile
    finite_vals = act_vol[np.isfinite(act_vol)]
    finite_vals = finite_vals[finite_vals != 0]
    if finite_vals.size > 0:
        vmax = np.percentile(np.abs(finite_vals), 98)
        vmax = float(max(vmax, 1e-6))
    else:
        vmax = 1.0
    vmin = -vmax
    thr = 0.15 * vmax  # soft threshold for visualization

    # 1) Stat map (orthoview) on bg
    disp1 = plotting.plot_stat_map(
        act_img, bg_img=bg_img, display_mode="ortho",
        threshold=thr, vmax=vmax, cmap="cold_hot",
        title=f"First stimulus activation ({source_tag}) — subj{sp}"
    )
    out_png1 = args.out_dir / "first_stim_activation_ortho.png"
    disp1.savefig(str(out_png1)); disp1.close()

    # 2) Glass brain
    disp2 = plotting.plot_glass_brain(
        act_img, display_mode="lzry", threshold=thr, vmax=vmax,
        cmap="cold_hot", title=f"Glass brain — first stimulus (subj{sp})"
    )
    out_png2 = args.out_dir / "first_stim_activation_glass.png"
    disp2.savefig(str(out_png2)); disp2.close()

    # 3) Used-voxel mask overlay
    disp3 = plotting.plot_roi(
        used_mask_img, bg_img=bg_img, display_mode="ortho",
        title=f"Used voxels mask (HDF5 selection) — subj{sp}"
    )
    out_png3 = args.out_dir / "used_voxels_mask_overlay.png"
    disp3.savefig(str(out_png3)); disp3.close()

    # 4) Distribution of activation values
    plt.figure(figsize=(6, 3.5))
    plt.hist(finite_vals, bins=60, color="#365CF5", alpha=0.85)
    plt.axvline(0, color="k", linewidth=1)
    plt.title(f"Activation distribution — first stimulus (subj{sp})", pad=10)
    plt.xlabel("Activation"); plt.ylabel("Voxel count")
    plt.tight_layout()
    out_png4 = args.out_dir / "first_stim_activation_hist.png"
    plt.savefig(out_png4, dpi=200); plt.close()

    print("[DONE]")
    print(" Saved:")
    print("  -", act_nii)
    print("  -", used_mask_nii)
    print("  -", out_png1)
    print("  -", out_png2)
    print("  -", out_png3)
    print("  -", out_png4)


if __name__ == "__main__":
    main()
