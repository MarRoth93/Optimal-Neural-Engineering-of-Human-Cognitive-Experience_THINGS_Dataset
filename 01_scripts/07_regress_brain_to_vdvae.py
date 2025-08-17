#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_regress_brain_to_vdvae.py

Map fMRI -> VDVAE latents with ridge regression (minimal changes from NSD version).

Defaults assume you ran:
- 05_map_ids_to_paths.py  -> created train/test image path lists
- 06_extract_vdvae_features.py -> saved things_vdvae_features_31l.npz
- 04_split_and_average_things.py -> produced train/test fMRI arrays

Usage (defaults for subj01):
python 07_regress_brain_to_vdvae.py --sub 1
"""

import sys
import os
import pickle
import argparse
import numpy as np
import sklearn.linear_model as skl

# ------------------ args ------------------
ap = argparse.ArgumentParser(description='Ridge regression: fMRI -> VDVAE latents (THINGS).')
ap.add_argument("-sub", "--sub", type=int, default=1, help="Subject number (e.g., 1)")
ap.add_argument("--alpha", type=float, default=70000, help="Ridge alpha (default 50k)")
ap.add_argument("--features_npz",
                help="Path to NPZ with train_latents/test_latents (VDVAE).")
ap.add_argument("--train_fmri",
                help="Path to train fMRI .npy")
ap.add_argument("--test_fmri",
                help="Path to test fMRI .npy")
ap.add_argument("--out_pred_dir",
                help="Dir for predicted latents .npy")
ap.add_argument("--out_weights_dir",
                help="Dir for regression weights .pkl")
args = ap.parse_args()

sub = int(args.sub)

# ------------------ defaults for THINGS layout ------------------
if args.features_npz is None:
    args.features_npz = f"/home/rothermm/THINGS/02_data/extracted_features/subj{sub:02d}/things_vdvae_features_31l.npz"

if args.train_fmri is None:
    args.train_fmri = f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sub:02d}/train_fmri.npy"

if args.test_fmri is None:
    args.test_fmri = f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sub:02d}/test_fmri.npy"

if args.out_pred_dir is None:
    args.out_pred_dir = f"/home/rothermm/THINGS/02_data/predicted_features/subj{sub:02d}"

if args.out_weights_dir is None:
    args.out_weights_dir = f"/home/rothermm/THINGS/02_data/regression_weights/subj{sub:02d}"

os.makedirs(args.out_pred_dir, exist_ok=True)
os.makedirs(args.out_weights_dir, exist_ok=True)

print("Subject:", sub)
print("Features:", args.features_npz)
print("Train fMRI:", args.train_fmri)
print("Test  fMRI:", args.test_fmri)
print("Alpha:", args.alpha)

# ------------------ load data ------------------
ft = np.load(args.features_npz)
train_latents = ft["train_latents"]
test_latents  = ft["test_latents"]

train_fmri = np.load(args.train_fmri)
test_fmri  = np.load(args.test_fmri)

# ------------------ preprocess fMRI (same as NSD) ------------------
train_fmri = train_fmri / 300.0
test_fmri  = test_fmri  / 300.0

norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
# guard against zero std
norm_scale_train[norm_scale_train == 0] = 1.0

train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri  = (test_fmri  - norm_mean_train) / norm_scale_train

print("train_fmri mean/std:", float(np.mean(train_fmri)), float(np.std(train_fmri)))
print("test_fmri  mean/std:", float(np.mean(test_fmri)),  float(np.std(test_fmri)))
print("train_fmri max/min:", float(np.max(train_fmri)), float(np.min(train_fmri)))
print("test_fmri  max/min:", float(np.max(test_fmri)),  float(np.min(test_fmri)))

# ------------------ ridge regression ------------------
print("Training Ridge regressor (fMRI -> VDVAE latents)...")
reg = skl.Ridge(alpha=args.alpha, max_iter=30000, fit_intercept=True)
reg.fit(train_fmri, train_latents)

pred_test_latent = reg.predict(test_fmri)

# match distribution to train_latents (same post-step as your NSD code)
std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent, axis=0)) / np.std(pred_test_latent, axis=0)
pred_latents = std_norm_test_latent * np.std(train_latents, axis=0) + np.mean(train_latents, axis=0)

score = reg.score(test_fmri, test_latents)
print("Test R^2:", float(score))

# ------------------ save ------------------
alpha_tag = (f"{int(args.alpha):d}" if float(args.alpha).is_integer() else f"{args.alpha}").replace(".", "p")
pred_path = os.path.join(args.out_pred_dir, f"things_vdvae_pred_sub{sub:02d}_31l_alpha{alpha_tag}.npy")
np.save(pred_path, pred_latents)
print("Saved predictions ->", pred_path)

datadict = {
    "weight": reg.coef_,        # shape: (latent_dim, n_voxels)
    "bias": reg.intercept_,     # shape: (latent_dim,)
    "alpha": args.alpha,
    "norm_mean_train": norm_mean_train,
    "norm_scale_train": norm_scale_train,
}
weights_path = os.path.join(args.out_weights_dir, "vdvae_regression_weights.pkl")
with open(weights_path, "wb") as f:
    pickle.dump(datadict, f)
print("Saved weights ->", weights_path)
