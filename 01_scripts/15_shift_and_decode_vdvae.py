#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_shift_and_decode_vdvae.py

Add alpha * theta to predicted test latents and decode images with VDVAE.

Defaults (subj01):
- Predicted latents  (tries DNN first, then ridge):
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
  /home/rothermm/THINGS/03_results/vdvae_shifted/subj01/{emonet|memnet}/alpha_{±k}/0.png ...

Usage:
python 15_shift_and_decode_vdvae.py --sub 1 --alphas -4 -3 -2 0 2 3 4
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import sys

# ---------- args ----------
ap = argparse.ArgumentParser(description="Shift predicted VDVAE latents by alpha*theta and decode.")
ap.add_argument("--sub", type=int, default=1, help="Subject number (e.g., 1)")
ap.add_argument("--pred_npy", type=Path, default=None, help="Predicted test latents .npy")
ap.add_argument("--feat_dir", type=Path, default=None, help="Dir with ref_latents.npz")
ap.add_argument("--theta_emo", type=Path, default=None, help="Path to EmoNet theta .npy")
ap.add_argument("--theta_mem", type=Path, default=None, help="Path to MemNet theta .npy")
ap.add_argument("--which", choices=["both","emonet","memnet"], default="both", help="Which theta(s) to use")
ap.add_argument("--alphas", type=float, nargs="+", default=[-4,-3,-2,0,2,3,4], help="Alpha values")
ap.add_argument("--bs", type=int, default=30, help="Batch size for decoding")
ap.add_argument("--vdvae_root", type=Path, default=Path("/home/rothermm/brain-diffuser/vdvae"))
ap.add_argument("--model_dir", type=Path, default=Path("/home/rothermm/brain-diffuser/vdvae/model"))
ap.add_argument("--out_base", type=Path, default=None, help="Base output dir for shifted decodes")
args = ap.parse_args()

sub = int(args.sub)
sp = f"{sub:02d}"

# ---------- defaults for THINGS layout ----------
if args.feat_dir is None:
    args.feat_dir = Path(f"/home/rothermm/THINGS/02_data/extracted_features/subj{sp}")
if args.out_base is None:
    args.out_base = Path(f"/home/rothermm/THINGS/03_results/vdvae_shifted/subj{sp}")
# find predicted latents if not given
if args.pred_npy is None:
    cand1 = Path(f"/home/rothermm/THINGS/02_data/predicted_features/subj{sp}/things_vdvae_pred_sub{sp}_31l_DNN_w2048_d6.npy")
    cand2 = Path(f"/home/rothermm/THINGS/02_data/predicted_features/subj{sp}/things_vdvae_pred_sub{sp}_31l_alpha50000.npy")
    args.pred_npy = cand1 if cand1.exists() else cand2

if args.theta_emo is None:
    args.theta_emo = Path(f"/home/rothermm/THINGS/03_results/thetas/subj{sp}/theta_emonet_decoded_top10_minus_bottom10.npy")
if args.theta_mem is None:
    args.theta_mem = Path(f"/home/rothermm/THINGS/03_results/thetas/subj{sp}/theta_memnet_decoded_top10_minus_bottom10.npy")

# ---------- imports from vdvae ----------
sys.path.append(str(args.vdvae_root))
import torch
from PIL import Image
from image_utils import *
from model_utils import *
from train_helpers import restore_params
from vae import VAE

print("Libs imported")

# ---------- model config (same as before) ----------
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

# ---------- load inputs ----------
if not args.pred_npy.exists():
    raise FileNotFoundError(f"Predicted latents not found: {args.pred_npy}")
ref_latent_path = args.feat_dir / "ref_latents.npz"
if not ref_latent_path.exists():
    raise FileNotFoundError(f"ref_latents.npz not found: {ref_latent_path}")

pred_latents = np.load(args.pred_npy)  # shape (N, D_flat)
ref_latent = np.load(ref_latent_path, allow_pickle=True)["ref_latent"]
N, D = pred_latents.shape
print(f"[INFO] pred_latents: {pred_latents.shape}")

need_emo = args.which in ("both","emonet")
need_mem = args.which in ("both","memnet")
if need_emo and not args.theta_emo.exists():
    raise FileNotFoundError(f"Emo theta missing: {args.theta_emo}")
if need_mem and not args.theta_mem.exists():
    raise FileNotFoundError(f"Mem theta missing: {args.theta_mem}")

theta_map = {}
if need_emo: theta_map["emonet"] = np.load(args.theta_emo)
if need_mem: theta_map["memnet"] = np.load(args.theta_mem)

# sanity on shapes
for k, th in theta_map.items():
    if th.shape[0] != D:
        raise ValueError(f"Theta shape mismatch for {k}: {th.shape} vs pred D={D}")

# ---------- latent pyramid helpers ----------
layer_dims = np.array([
    2**4,2**4, 2**8,2**8,2**8,2**8,
    2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,
    2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,
    2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,
    2**14
], dtype=np.int64)
offs = np.cumsum(np.r_[0, layer_dims])
num_latents = len(layer_dims)

def flat_to_pyr_subset(latents_flat: np.ndarray, idxs):
    """Return list of 31 arrays, each [B, C, H, W] for batch indices idxs."""
    B = len(idxs)
    out = []
    for i in range(num_latents):
        seg = latents_flat[np.ix_(idxs, np.arange(offs[i], offs[i+1]))]
        c, h, w = ref_latent[i]['z'].shape[1:]
        out.append(seg.reshape(B, c, h, w))
    return out

# ---------- decode loop ----------
for assessor_name, theta in theta_map.items():
    for a in args.alphas:
        a_dir = args.out_base / assessor_name / f"alpha_{a:+g}"
        a_dir.mkdir(parents=True, exist_ok=True)

        # shift or not
        if float(a) == 0.0:
            latents_shifted = pred_latents
        else:
            latents_shifted = pred_latents + float(a) * theta

        # batches
        B = int(args.bs)
        nb = (N + B - 1) // B
        print(f"[{assessor_name}] alpha={a:+g} | N={N}  batches={nb} -> {a_dir}")

        for bi in range(nb):
            lo = bi * B
            hi = min((bi + 1) * B, N)
            ids = list(range(lo, hi))

            # build per-level tensors on device
            pyr = flat_to_pyr_subset(latents_shifted, ids)
            batch_levels = [torch.from_numpy(x).float().to(device, non_blocking=True) for x in pyr]

            with torch.no_grad():
                px_z = ema_vae.decoder.forward_manual_latents(len(ids), batch_levels, t=None)
                generated = ema_vae.decoder.out_net.sample(px_z)  # list of HWC uint8

            for j, arr in enumerate(generated):
                img = Image.fromarray(arr)
                img = img.resize((512, 512), resample=Image.BICUBIC)
                img.save(a_dir / f"{lo + j}.png")

            print(f"  saved [{lo}:{hi})")

        # small meta per alpha
        meta = {
            "subject": sub,
            "assessor": assessor_name,
            "alpha": float(a),
            "pred_npy": str(args.pred_npy),
            "theta_path": str(args.theta_emo if assessor_name=='emonet' else args.theta_mem),
            "N": int(N),
        }
        with open(a_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

print(f"[DONE] Wrote shifted decodes under: {args.out_base}")
