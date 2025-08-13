#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_decode_vdvae_from_pred.py

Decode images from predicted VDVAE latents for THINGS.
Minimal changes vs. your NSD decoder; only paths + small robustness tweaks.

Inputs (defaults for subj01):
- Predicted latents:  /home/rothermm/THINGS/02_data/predicted_features/subj01/things_vdvae_pred_sub01_31l_alpha50000.npy
- Reference latents:  /home/rothermm/THINGS/02_data/extracted_features/subj01/ref_latents.npz
- VDVAE checkpoints:  /home/rothermm/brain-diffuser/vdvae/model

Output:
- PNGs in /home/rothermm/THINGS/03_results/vdvae/subj01/

Usage:
python 08_decode_vdvae_from_pred.py --sub 1 --alpha 50000 --bs 30
"""

import sys, os, argparse
import numpy as np
from pathlib import Path

# -------------------- args --------------------
ap = argparse.ArgumentParser(description="Decode images from predicted VDVAE latents (THINGS).")
ap.add_argument("--sub", type=int, default=1, help="Subject number (e.g., 1)")
ap.add_argument("--alpha", type=float, default=50000, help="Alpha used for ridge (filename tag)")
ap.add_argument("--bs", type=int, default=30, help="Batch size for decoding")
ap.add_argument("--vdvae_root", type=Path, default=Path("/home/rothermm/brain-diffuser/vdvae"))
ap.add_argument("--model_dir", type=Path, default=Path("/home/rothermm/brain-diffuser/vdvae/model"))
ap.add_argument("--pred_dir", type=Path, default=None, help="Dir with predicted latents .npy")
ap.add_argument("--feat_dir", type=Path, default=None, help="Dir with ref_latents.npz")
ap.add_argument("--out_dir", type=Path, default=None, help="Where to save decoded PNGs")
args = ap.parse_args()

sub = int(args.sub)
sub_pad = f"{sub:02d}"

# -------------------- default paths (THINGS) --------------------
if args.pred_dir is None:
    args.pred_dir = Path(f"/home/rothermm/THINGS/02_data/predicted_features/subj{sub_pad}")
if args.feat_dir is None:
    args.feat_dir = Path(f"/home/rothermm/THINGS/02_data/extracted_features/subj{sub_pad}")
if args.out_dir is None:
    args.out_dir = Path(f"/home/rothermm/THINGS/03_results/vdvae/subj{sub_pad}")

alpha_tag = (f"{int(args.alpha):d}" if float(args.alpha).is_integer() else f"{args.alpha}").replace(".", "p")
pred_latents_path = args.pred_dir / f"things_vdvae_pred_sub{sub_pad}_31l_alpha{alpha_tag}.npy"
ref_latent_path   = args.feat_dir / "ref_latents.npz"

print("Pred path:", pred_latents_path)
print("Ref path :", ref_latent_path)


# -------------------- imports from vdvae --------------------
sys.path.append(str(args.vdvae_root))
import torch
from PIL import Image
import torchvision.transforms as T  # (kept for parity, not strictly needed)

from image_utils import *
from model_utils import *
from train_helpers import restore_params
from vae import VAE

print("Libs imported")

# -------------------- model config (same as your NSD code) --------------------
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

# -------------------- load predicted + ref latents --------------------
if not pred_latents_path.exists():
    raise FileNotFoundError(f"Predicted latents not found: {pred_latents_path}")
if not ref_latent_path.exists():
    raise FileNotFoundError(f"ref_latents.npz not found: {ref_latent_path}")

pred_latents = np.load(pred_latents_path)  # shape (N, sum(layer_dims))
print(f"Loaded predicted latents: {pred_latents.shape}")
ref_latent = np.load(ref_latent_path, allow_pickle=True)["ref_latent"]
print(f"Loaded ref_latent: {ref_latent_path}")

# -------------------- reshape flat vectors to per-level feature maps --------------------
# layer_dims must match your extraction stacking order
layer_dims = np.array([
    2**4,2**4,2**8,2**8,2**8,
    2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,
    2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,
    2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,
    2**12,2**14
], dtype=np.int64)
num_latents = len(layer_dims)

def flat_to_pyramid(latents_flat, ref):
    """Split/reshape (N, sum(layer_dims)) -> list of NCHW tensors per level."""
    out = []
    offs = np.cumsum(np.r_[0, layer_dims])
    for i in range(num_latents):
        t_lat = latents_flat[:, offs[i]:offs[i+1]]
        # Use ref shapes for (C,H,W)
        c, h, w = ref[i]['z'].shape[1:]
        out.append(t_lat.reshape(len(latents_flat), c, h, w))
    return out

pyr_latents = flat_to_pyramid(pred_latents, ref_latent)

# -------------------- decode in batches --------------------
args.out_dir.mkdir(parents=True, exist_ok=True)
N = pred_latents.shape[0]
B = int(args.bs)
num_batches = (N + B - 1) // B
print(f"Decoding {N} images in {num_batches} batches -> {args.out_dir}")

for bi in range(num_batches):
    lo = bi * B
    hi = min((bi + 1) * B, N)
    ids = list(range(lo, hi))
    # collect batch tensors per level on device
    batch_levels = []
    for lvl in range(num_latents):
        x = torch.from_numpy(pyr_latents[lvl][ids]).float().to(device, non_blocking=True)
        batch_levels.append(x)
    with torch.no_grad():
        px_z = ema_vae.decoder.forward_manual_latents(len(ids), batch_levels, t=None)
        generated = ema_vae.decoder.out_net.sample(px_z)  # uint8 HWC per sample
    # save
    for j, img_arr in enumerate(generated):
        img = Image.fromarray(img_arr)
        img = img.resize((512, 512), resample=Image.BICUBIC)
        img.save(args.out_dir / f"{lo + j}.png")
    print(f"  batch {bi+1}/{num_batches} saved [{lo}:{hi})")

print(f"[DONE] Images written to: {args.out_dir}")
