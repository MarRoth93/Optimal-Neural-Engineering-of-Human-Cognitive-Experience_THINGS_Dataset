#!/bin/bash
#SBATCH --job-name=vdvae_feats800_s01
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/06_vdvae_feats800_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/06_vdvae_feats800_s01_%j.err

echo "==== Job started on $(hostname) at $(date) ===="

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"
mkdir -p /home/rothermm/THINGS/01_scripts/logs

# Input paths (800split)
TRAIN_PATHS="/home/rothermm/THINGS/02_data/preprocessed_data/subj01/800split/train_image_paths.txt"
TEST_PATHS="/home/rothermm/THINGS/02_data/preprocessed_data/subj01/800split/test_image_paths.txt"
test -f "$TRAIN_PATHS" || { echo "Missing: $TRAIN_PATHS" >&2; exit 1; }
test -f "$TEST_PATHS"  || { echo "Missing: $TEST_PATHS"  >&2; exit 1; }

# VDVAE model + repo
VDVAE_ROOT="/home/rothermm/brain-diffuser/vdvae"
MODEL_DIR="/home/rothermm/brain-diffuser/vdvae/model"

# Output dir for features (separate from 100-test)
OUTDIR="/home/rothermm/THINGS/02_data/extracted_features/subj01/800split"
mkdir -p "$OUTDIR"

# Run extraction
CMD=( python -u 06_extract_vdvae_features.py
  --vdvae_root "$VDVAE_ROOT"
  --model_dir "$MODEL_DIR"
  --train_paths "$TRAIN_PATHS"
  --test_paths "$TEST_PATHS"
  --out_dir "$OUTDIR"
  --bs 128
  --num_latents 31
  --num_workers 8
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/06_vdvae_feats800_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
