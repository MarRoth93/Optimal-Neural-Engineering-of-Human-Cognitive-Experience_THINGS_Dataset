# Optimal Neural Engineering of Human Cognitive Experience – THINGS Extension

This repository extends [MarRoth93/Optimal-Neural-Engineering-of-Human-Cognitive-Experience](https://github.com/MarRoth93/Optimal-Neural-Engineering-of-Human-Cognitive-Experience) by attempting to replicate its neural decoding results using the **THINGS** dataset instead of **NSD**.

## Repository layout

- `01_scripts/` – Python utilities for data preparation, feature extraction, regression, and evaluation.
- `03_slurm_scripts/` – SLURM job scripts corresponding to each Python script.

## Script overview

### 01_scripts

- `01_make_vc_mask_voxelmeta.py` – Loads voxel metadata and a subject‑specific brain mask, flags rows belonging to visual cortex ROIs, converts their voxel coordinates into a binary volume, and writes both the mask and an overlay image.
- `02_extract_betas_to_numpy.py` – Iterates over single‑trial beta NIfTIs, resamples the VC mask if necessary, extracts the masked voxel values into a memory‑mapped trial×voxel matrix, records basic metadata, and optionally checks the number of events.
- `03_extract_h5_to_numpy_v2.py` – Selects visual‑cortex voxel IDs from metadata, indexes the large HDF5 response matrix to load only those rows, optionally z‑scores the time courses, and stores the resulting array together with index/voxel‑id sidecars.
- `04_split_and_average_things.py` – Joins the voxel matrix with stimulus metadata, separates train and test trials by `trial_type`, averages repeated presentations in the test set (and optionally in train), and saves index files, ID lists, and a JSON summary.
- `05_map_ids_to_paths.py` – Recursively indexes the image root, handles duplicate filenames, and translates lists of THINGS image IDs into absolute file paths for later feature extraction.
- `06_extract_vdvae_features.py` – Loads a pretrained VDVAE, reads train/test image path lists, converts images to 64×64, and stacks hierarchical latent vectors for both splits along with a reference latent.
- `07_regress_brain_to_vdvae.py` – Normalizes fMRI matrices, fits a ridge regression that predicts VDVAE latents from brain responses, outputs the predicted test latents, and stores the regression weights and normalization parameters.
- `07b_train_deep_regressor.py` – Standardizes data and trains a deep residual MLP using cosine and MSE losses with validation‑based early stopping, then de‑standardizes and saves the predicted latents plus a training summary.
- `08_decode_vdvae_from_pred.py` – Reshapes flat predicted latent vectors into per‑level feature maps, feeds them through the VDVAE decoder in batches, and saves the reconstructed images as PNGs.
- `09_compare_original_vs_decoded.py` – Loads original test images and their reconstructions, resizes them to a common size, and assembles a side‑by‑side comparison figure for the first N examples.
- `10_compute_assessor_scores.py` – Uses EmoNet and MemNet to score original and decoded images, records the score vectors, and produces line and scatter plots summarizing agreement between them.
- `11_visualize_first_stim_activation.py` – Reconstructs a 3D activation map for the first stimulus using voxel coordinates, writes NIfTI volumes for the activation and mask, and generates orthographic, glass‑brain, and histogram visualizations.
- `12_caption_images.py` – Batch‑generates captions for train and test image lists with BLIP or BLIP‑2, supports optional prompts and beam search, and writes TSV/JSONL/TXT files aligned with the image order with resume support.
- `13_plot_test_captions.py` – Plots the first N test images with their BLIP‑2 captions by robustly matching paths and saves a composite figure.
- `14_compute_theta_from_scores.py` – Contrasts top and bottom fractions of EmoNet and MemNet scores to form theta vectors in latent space and records summary statistics.
- `15_shift_and_decode_vdvae.py` – Adds scaled theta vectors to predicted latents, decodes them with VDVAE, and writes the resulting shifted reconstructions.

## Acknowledgments

This work builds on the original repository's methodology and adapts it for the THINGS dataset.

