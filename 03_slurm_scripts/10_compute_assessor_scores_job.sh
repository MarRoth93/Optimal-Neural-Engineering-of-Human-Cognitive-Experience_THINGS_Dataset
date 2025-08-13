#!/bin/bash
#SBATCH --job-name=vdvaeAssess_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/vdvaeAssess_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/vdvaeAssess_s01_%j.err

set -euo pipefail

echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

TEST_PATHS="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/test_image_paths.txt"
DECODED_DIR="/home/rothermm/THINGS/03_results/vdvae/subj${SUBJ_PAD}"
ASSESSORS_ROOT="/home/rothermm/brain-diffuser/assessors"
SCORES_DIR="/home/rothermm/THINGS/03_results/assessor_scores/subj${SUBJ_PAD}"
PLOTS_DIR="/home/rothermm/THINGS/03_results/plots"

# Conda env (your working method)
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V

# Checks
[[ -e "$TEST_PATHS" ]] || { echo "Missing test paths: $TEST_PATHS" >&2; exit 1; }
[[ -d "$DECODED_DIR" ]] || { echo "Missing decoded dir: $DECODED_DIR" >&2; exit 1; }
[[ -e "${ASSESSORS_ROOT}/image_mean.npy" ]] || { echo "Missing MemNet mean: ${ASSESSORS_ROOT}/image_mean.npy" >&2; exit 1; }

CMD=( python -u 10_compute_assessor_scores.py
  --sub "$SUBJ"
  --test_paths "$TEST_PATHS"
  --decoded_dir "$DECODED_DIR"
  --assessors_root "$ASSESSORS_ROOT"
  --scores_dir "$SCORES_DIR"
  --plots_dir "$PLOTS_DIR"
)

# change the run line to this (note 2>&1):
echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "/home/rothermm/THINGS/01_scripts/logs/vdvaeAssess_s01_${SLURM_JOB_ID}.debug.log"


echo "==== Job finished at $(date) ===="
