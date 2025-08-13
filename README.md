# Optimal Neural Engineering of Human Cognitive Experience – THINGS Extension

This repository extends [MarRoth93/Optimal-Neural-Engineering-of-Human-Cognitive-Experience](https://github.com/MarRoth93/Optimal-Neural-Engineering-of-Human-Cognitive-Experience) by attempting to replicate its neural decoding results using the **THINGS** dataset instead of **NSD**.

## Repository layout

- `01_scripts/` – Python utilities for data preparation, feature extraction, regression, and evaluation.
- `03_slurm_scripts/` – SLURM job scripts corresponding to each Python script.

## Script overview

### 01_scripts

- `01_make_vc_mask_voxelmeta.py` – Build a visual cortex mask from voxel metadata and save NIfTI and overlay images.
- `02_extract_betas_to_numpy.py` – Apply the mask to single-trial NIfTI betas and stack them into a trial × voxel matrix (`X.npy`).
- `03_extract_h5_to_numpy_v2.py` – Load HDF5 voxel responses and save selected VC voxels into `X.npy` with metadata.
- `04_split_and_average_things.py` – Split trials into train/test sets and average repeated test trials; saves arrays and metadata.
- `05_map_ids_to_paths.py` – Map THINGS image IDs to absolute file paths, handling subfolders and duplicates.
- `06_extract_vdvae_features.py` – Extract hierarchical VDVAE latents for train/test image sets.
- `07_regress_brain_to_vdvae.py` – Ridge regression from fMRI responses to VDVAE latents and save predictions/weights.
- `07b_train_deep_regressor.py` – Train a deep MLP regressor with residual blocks to predict VDVAE latents from fMRI.
- `08_decode_vdvae_from_pred.py` – Decode images from predicted VDVAE latents and save them as PNGs.
- `09_compare_original_vs_decoded.py` – Create side‑by‑side figures comparing original test images and their reconstructions.
- `10_compute_assessor_scores.py` – Score original and decoded images with EmoNet and MemNet assessors and plot results.
- `11_visualize_first_stim_activation.py` – Visualize the first stimulus activation pattern and the used voxel mask.
- `12_caption_images.py` – Generate captions for train/test images using BLIP or BLIP‑2 models.

## Acknowledgments

This work builds on the original repository's methodology and adapts it for the THINGS dataset.

