#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_extract_vdvae_features.py

Extract VDVAE latents for THINGS images listed in:
- train_image_paths.txt
- test_image_paths.txt

The code mirrors your NSD pipeline with minimal edits:
- Replaces the npy-based dataset with a path-list dataset (PIL -> 64x64)
- Keeps set_up_data(), load_vaes(), preprocess_fn, EMA VAE, and latent stacking

Usage:
python 06_extract_vdvae_features.py \
  --vdvae_root /home/rothermm/brain-diffuser/vdvae \
  --model_dir  /home/rothermm/brain-diffuser/vdvae/model \
  --train_paths /home/rothermm/THINGS/02_data/preprocessed_data/subj01/train_image_paths.txt \
  --test_paths  /home/rothermm/THINGS/02_data/preprocessed_data/subj01/test_image_paths.txt \
  --out_dir     /home/rothermm/THINGS/02_data/extracted_features/subj01 \
  --bs 128 --num_latents 31
"""

import sys, os, argparse, socket, json, subprocess, pickle
from pathlib import Path

# ----------------- args -----------------
ap = argparse.ArgumentParser()
ap.add_argument("--vdvae_root", type=Path, required=True,
               help="Path to your vdvae repo (so we can import modules)")
ap.add_argument("--model_dir", type=Path, required=True,
               help="Path containing the pretrained VDVAE checkpoints")
ap.add_argument("--train_paths", type=Path, required=True,
               help="Text file with absolute image paths (train)")
ap.add_argument("--test_paths", type=Path, required=True,
               help="Text file with absolute image paths (test)")
ap.add_argument("--out_dir", type=Path, required=True,
               help="Output directory for npz files (features, ref_latents)")
ap.add_argument("--bs", type=int, default=30, help="Batch size")
ap.add_argument("--num_latents", type=int, default=31, help="Number of hierarchical latents to extract")
ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
args = ap.parse_args()

# ----------------- imports from vdvae -----------------
sys.path.append(str(args.vdvae_root))

import torch
import numpy as np
from hps import Hyperparams
from utils import logger
from data import mkdir_p
from vae import VAE
from image_utils import *
from model_utils import *
from train_helpers import restore_params

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

print('Libs imported')

# ----------------- VDVAE setup (as in your NSD code) -----------------
H = {
    'image_size': 64, 'image_channels': 3, 'seed': 0, 'port': 29500,
    'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test',
    'hparam_sets': 'imagenet64',
    'restore_path':        str(args.model_dir / 'imagenet64-iter-1600000-model.th'),
    'restore_ema_path':    str(args.model_dir / 'imagenet64-iter-1600000-model-ema.th'),
    'restore_log_path':    str(args.model_dir / 'imagenet64-iter-1600000-log.jsonl'),
    'restore_optimizer_path': str(args.model_dir / 'imagenet64-iter-1600000-opt.th'),
    'dataset': 'imagenet64',
    'ema_rate': 0.999,
    'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5',
    'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12',
    'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25,
    'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True,
    'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0,
    'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0,
    'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9,
    'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000,
    'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1,
    'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8,
    'num_variables_visualize': 6, 'num_temperatures_visualize': 3,
    'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'
}

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

H = dotdict(H)

# set_up_data gives you preprocess_fn exactly as in your NSD code
H, preprocess_fn = set_up_data(H)

print('Model is Loading')
ema_vae = load_vaes(H)  # EMA model

# ----------------- Dataset: list of absolute paths -----------------
def _read_paths(txt: Path):
    with open(txt, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]

class ImageListDataset(Dataset):
    """
    Minimal change vs. your NSD dataset:
    - Reads paths from a txt file (one absolute path per line)
    - Loads with PIL, converts to RGB, resizes to 64x64
    - Returns float tensor with HWC layout, same as you used
    """
    def __init__(self, paths_file: Path):
        self.paths = _read_paths(paths_file)
        self.resize = T.Resize((64, 64), interpolation=T.InterpolationMode.BILINEAR)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = self.resize(img)
        arr = np.array(img, dtype=np.float32)  # HWC float
        return torch.from_numpy(arr)
    def __len__(self):
        return len(self.paths)

# ----------------- DataLoaders -----------------
bs = int(args.bs)
train_ds = ImageListDataset(args.train_paths)
test_ds  = ImageListDataset(args.test_paths)

trainloader = DataLoader(train_ds, batch_size=bs, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
testloader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

num_latents = int(args.num_latents)
out_dir = Path(args.out_dir)
(out_dir).mkdir(parents=True, exist_ok=True)

# ----------------- Extract test latents (save ref once) -----------------
test_latents = []
ref_latent_saved = False

for i, x in enumerate(testloader):
    data_input, target = preprocess_fn(x)  # unchanged
    with torch.no_grad():
        print(f"[test] processed: {i * bs}")
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)

        if not ref_latent_saved:
            np.savez(out_dir / "ref_latents.npz", ref_latent=stats)
            print(f"ref_latent saved -> {out_dir/'ref_latents.npz'}")
            ref_latent_saved = True

        batch_latent = []
        for j in range(num_latents):
            # stats[j]['z']: (B, C, H, W) -> flatten per sample
            batch_latent.append(stats[j]['z'].cpu().numpy().reshape(len(data_input), -1))
        test_latents.append(np.hstack(batch_latent))

test_latents = np.concatenate(test_latents, axis=0)

# ----------------- Extract train latents -----------------
train_latents = []
for i, x in enumerate(trainloader):
    data_input, target = preprocess_fn(x)
    with torch.no_grad():
        print(f"[train] processed: {i * bs}")
        activations = ema_vae.encoder.forward(data_input)
        px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)

        batch_latent = []
        for j in range(num_latents):
            batch_latent.append(stats[j]['z'].cpu().numpy().reshape(len(data_input), -1))
        train_latents.append(np.hstack(batch_latent))

train_latents = np.concatenate(train_latents, axis=0)

# ----------------- Save -----------------
np.savez(out_dir / "things_vdvae_features_31l.npz",
         train_latents=train_latents,
         test_latents=test_latents)

print(f"[DONE] train {train_latents.shape}, test {test_latents.shape} -> {out_dir/'things_vdvae_features_31l.npz'}")
