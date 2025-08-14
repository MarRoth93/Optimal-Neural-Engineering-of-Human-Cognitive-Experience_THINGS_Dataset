#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_shift_and_decode_vdvae.py

Add alpha * theta to predicted test latents and decode images with VDVAE.

Key points / fixes:
- Prints the EXACT predicted-latents file being used.
- Supports two alpha modes:
    * raw   : delta = a * theta
    * sigma : delta = (a * std_along_theta) * unit(theta)   [safer default]
- Uses the same VDVAE loading path as 08_decode_vdvae_from_pred.py.
- Checks that sum(layer_dims) == latent dimension (shape safety).

Defaults (subj01):
- Predicted latents (if --pred_npy not given, tries these in order):
    /home/rothermm/THINGS/02_data/predicted_features/subj01/things_vdvae_pred_sub01_31l_DNN_w2048_d6.npy
    /home/rothermm/THINGS/02_data/predicted_features/subj01/things_vdvae_pred_sub01_31l_alpha50000.npy
- Ref latents (shapes):
    /home/rothermm/THINGS/02_data/extracted_features/subj01/ref_latents.npz
- Thetas (decoded, top10 - bottom10):
    /home/rothermm/THINGS/03_results/thetas/subj01/theta_emonet_decoded_top10_minus_bottom10.npy
    /home/rothermm/THINGS/03_results/thetas/subj01/theta_memnet_decoded_top10_minus_bottom10.npy
- VDVAE checkpoints:
    /home/rothermm/brain-diffuser/vdvae/model
- Output images:
    /home/rothermm/THINGS/03_results/vdvae_shifted/subj01/{emonet|memnet}/alpha_{±x}/0.png ...

Usage (safe σ mode with small steps):
python 15_shift_and_decode_vdvae.py --sub 1 --alpha_mode sigma --alphas -1 -0.5 0 0.5 1
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# -------------------- args --------------------
ap = argparse.ArgumentParser(description="Shift predicted VDVAE latents by alpha*theta and decode.")
ap.add_argument("--sub", type=int, default=1, help="Subject number (e.g., 1)")
ap.add_argument("--pred_npy", type=Path, default=None, help="Predicted test latents .npy")
ap.add_argument("--feat_dir", type=Path, default=None, help="Dir with ref_latents.npz")
ap.add_argument("--theta_emo", type=Path, default=None, help="Path to EmoNet theta .npy")
ap.add_argument("--theta_mem", type=Path, default=None, help="Path to MemNet theta .npy")
ap.add_argument("--which", choices=["both","emonet","memnet"], default="both", help="Which theta(s) to use")
ap.add_argument("--alphas", type=float, nargs="+", default=[-1, -0.5, 0, 0.5, 1], help="Alpha values")
ap.add_argument("--alpha_mode", choices=["raw","sigma"], default="sigma",
                help="raw: delta = a*theta;  sigma: delta = (a * std_along_theta) * unit(theta)")
ap.add_argument("--bs", type=int, default=30, help="Batch size for decoding")

# VDVAE paths
ap.add_argument("--vdvae_root", type=Path, default=Path("/home/rothermm/brain-diffuser/vdvae"))
ap.add_argument("--model_dir", type=Path, default=Path("/home/rothermm/brain-diffuser/vdvae/model"))

# Outputs
ap.add_argument("--out_base", type=Path, default=None, help="Base output dir for shifted decodes")
ap.add_argument("--debug_first_diff", action="store_true", help="Print mean|Δpixel| vs alpha=0 for index 0")
args = ap.parse_args()

sub = int(args.sub)
sp = f"{sub:02d}"

# -------------------- default THINGS layout --------------------
if args.feat_dir is None:
    args.feat_dir = Path(f"/home/rothermm/THINGS/02_data/extracted_features/subj{sp}")
if args.out_base is None:
    args.out_base = Path(f"/home/rothermm/THINGS/03_results/vdvae_shifted/subj{sp}")

# Resolve predicted latents if not passed
if args.pred_npy is None:
    cand1 = Path(f"/home/rothermm/THINGS/02_data/predicted_features/subj{sp}/things_vdvae_pred_sub{sp}_31l_DNN_w2048_d6.npy")
    cand2 = Path(f"/home/rothermm/THINGS/02_data/predicted_features/subj{sp}/things_vdvae_pred_sub{sp}_31l_alpha50000.npy")
    if cand1.exists():
        args.pred_npy = cand1
    else:
        args.pred_npy = cand2

if args.theta_emo is None:
    args.theta_emo = Path(f"/home/rothermm/THINGS/03_results/thetas/subj{sp}/theta_emonet_decoded_top10_minus_bottom10.npy")
if args.theta_mem is None:
    args.theta_mem = Path(f"/home/rothermm/THINGS/03_results/thetas/subj{sp}/theta_memnet_decoded_top10_minus_bottom10.npy")

# -------------------- imports from vdvae --------------------
sys.path.append(str(args.vdvae_root))
import torch
from PIL import Image
from image_utils import *        # provides set_up_data / load_vaes helpers in this codebase
from model_utils import *
from train_helpers import restore_params
from vae import VAE

print("Libs imported")

# -------------------- Model config (same as your working 08_ script) --------------------
H = {
    'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500,
    'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test',
    'hparam_sets': 'imagenet64',
    'restore_path':        str(args.model_dir / 'imagenet64-iter-1600000-model.th'),
    'restore_ema_path':    str(args.model_dir / 'imagenet64-iter-1600000-model-ema.th'),
    'restore_log_path':    str(args.model_dir / 'imagenet64-iter-1600000-log.jsonl'),
    'restore_optimizer_path': str(args.model_dir / 'imagenet64-iter-1600000-opt.th'),
    'dataset': 'imagenet64', 'ema_rate': 0.999,
    'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
    'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12',
    'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25,
    'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100,
    'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015,
    'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4,
    'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0,
    'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000,
    'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None,
    'epochs_per_eval_save': 1, 'num_images_visualize': 8,
    'num_variables_visualize': 6, 'num_temperatures_visualize': 3,
    'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'
}
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)
H, preprocess_fn = set_up_data(H)

print('Model is loading...')
ema_vae = load_vaes(H)  # EMA model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------- Load inputs & sanity --------------------
if not args.pred_npy.exists():
    raise FileNotFoundError(f"Predicted latents not found: {args.pred_npy}")
ref_latent_path = args.feat_dir / "ref_latents.npz"
if not ref_latent_path.exists():
    raise FileNotFoundError(f"ref_latents.npz not found: {ref_latent_path}")

print(f"[INFO] Using predicted latents file: {args.pred_npy}")
pred_latents = np.load(args.pred_npy)  # shape (N, D)
ref_latent = np.load(ref_latent_path, allow_pickle=True)["ref_latent"]

N, D = pred_latents.shape
print(f"[INFO] pred_latents shape: {pred_latents.shape}")

need_emo = args.which in ("both","emonet")
need_mem = args.which in ("both","memnet")
if need_emo and not args.theta_emo.exists():
    raise FileNotFoundError(f"EmoNet theta missing: {args.theta_emo}")
if need_mem and not args.theta_mem.exists():
    raise FileNotFoundError(f"MemNet theta missing: {args.theta_mem}")

# -------------------- Theta loading & normalization --------------------
theta_map = {}
if need_emo: theta_map["emonet"] = np.load(args.theta_emo)
if need_mem: theta_map["memnet"] = np.load(args.theta_mem)

for k, th in theta_map.items():
    if th.shape[0] != D:
        raise ValueError(f"Theta shape mismatch for {k}: {th.shape} vs latent dim {D}")
    nrm = float(np.linalg.norm(th))
    if nrm == 0:
        raise ValueError(f"Theta {k} has zero norm.")
    theta_map[k] = {"vec": th.astype(np.float32), "unit": (th / nrm).astype(np.float32)}
    print(f"[STATS] {k} theta L2={nrm:.4f}")

# Optional σ scaling (safe default)
theta_sigma = {}
if args.alpha_mode == "sigma":
    for k, pack in theta_map.items():
        u = pack["unit"]                       # (D,)
        proj = pred_latents @ u                # (N,)
        s = float(np.std(proj, ddof=1))
        if s == 0: s = 1.0
        theta_sigma[k] = s
        print(f"[STATS] sigma along {k} direction: {s:.6f}")

# -------------------- Latent pyramid helpers --------------------
# Layer dims MUST match how you stacked latents (same as 08_ script)
layer_dims = np.array([
    2**4,2**4,2**8,2**8,2**8,
    2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,
    2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,
    2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,
    2**12,2**14
], dtype=np.int64)
assert layer_dims.sum() == D, f"Sum(layer_dims)={layer_dims.sum()} != latent dim D={D}"
offs = np.cumsum(np.r_[0, layer_dims])
num_latents = len(layer_dims)

def flat_to_pyr_subset(latents_flat: np.ndarray, idxs):
    """Return list of 31 arrays, each [B, C, H, W] for batch indices idxs."""
    B = len(idxs)
    out = []
    for i in range(num_latents):
        seg = latents_flat[np.ix_(idxs, np.arange(offs[i], offs[i+1]))]   # (B, Ci*Hi*Wi)
        c, h, w = ref_latent[i]['z'].shape[1:]
        out.append(seg.reshape(B, c, h, w))
    return out

# For debug diffs vs alpha=0 (optional)
alpha0_cache = {}

# -------------------- Decode loop --------------------
args.out_base.mkdir(parents=True, exist_ok=True)

for assessor_name, pack in theta_map.items():
    u = pack["unit"]              # unit direction (D,)
    sigma_val = theta_sigma.get(assessor_name, 1.0)

    for a in args.alphas:
        a_dir = args.out_base / assessor_name / f"alpha_{a:+g}"
        a_dir.mkdir(parents=True, exist_ok=True)

        # Shift
        if float(a) == 0.0:
            latents_shifted = pred_latents
        else:
            if args.alpha_mode == "raw":
                delta = (float(a) * pack["vec"]).astype(np.float32)         # (D,)
            else:  # sigma mode
                delta = (float(a) * sigma_val * u).astype(np.float32)       # (D,)
            latents_shifted = pred_latents + delta                           # broadcast add over rows

        # Decode in batches
        B = int(args.bs)
        nb = (N + B - 1) // B
        print(f"[{assessor_name}] alpha={a:+g} | N={N}  batches={nb} -> {a_dir}")

        for bi in range(nb):
            lo = bi * B
            hi = min((bi + 1) * B, N)
            ids = list(range(lo, hi))

            pyr = flat_to_pyr_subset(latents_shifted, ids)
            batch_levels = [torch.from_numpy(x).float().to(device, non_blocking=True) for x in pyr]

            with torch.no_grad():
                px_z = ema_vae.decoder.forward_manual_latents(len(ids), batch_levels, t=None)
                generated = ema_vae.decoder.out_net.sample(px_z)  # list of HWC uint8

            # Save images
            for j, arr in enumerate(generated):
                img = Image.fromarray(arr)
                img = img.resize((512, 512), resample=Image.BICUBIC)
                img.save(a_dir / f"{lo + j}.png")

            # Optional quick diff vs alpha 0 for the very first image index 0
            if args.debug_first_diff and float(a) != 0.0 and lo == 0:
                # cache alpha 0 image
                if "alpha0" not in alpha0_cache:
                    # load alpha +0 version written earlier if exists
                    a0_path = args.out_base / assessor_name / "alpha_+0" / "0.png"
                    if a0_path.exists():
                        alpha0_cache["alpha0"] = np.asarray(Image.open(a0_path).convert("RGB"))
                if "alpha0" in alpha0_cache:
                    arr0 = alpha0_cache["alpha0"].astype(np.float32)
                    arrA = np.asarray(Image.open(a_dir / "0.png").convert("RGB")).astype(np.float32)
                    mdiff = float(np.mean(np.abs(arrA - arr0)))
                    print(f"    [DIFF] vs alpha_+0, idx 0: mean|Δpixel|={mdiff:.3f}")

        # Save small meta per alpha
        meta = {
            "subject": sub,
            "assessor": assessor_name,
            "alpha": float(a),
            "alpha_mode": args.alpha_mode,
            "pred_npy": str(args.pred_npy),
            "theta_path": str(args.theta_emo if assessor_name=='emonet' else args.theta_mem),
            "N": int(N),
        }
        with open(a_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

print(f"[DONE] Wrote shifted decodes under: {args.out_base}")
