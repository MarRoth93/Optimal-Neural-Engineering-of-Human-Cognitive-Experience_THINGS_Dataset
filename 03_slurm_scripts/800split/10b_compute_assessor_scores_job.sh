#!/bin/bash
#SBATCH --job-name=vdvaeAssess800_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/10_vdvaeAssess800_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/10_vdvaeAssess800_s01_%j.err

set -euo pipefail

echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

TEST_PATHS="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/800split/test_image_paths.txt"
DECODED_DIR="/home/rothermm/THINGS/03_results/vdvae/subj${SUBJ_PAD}/800split"
ASSESSORS_ROOT="/home/rothermm/brain-diffuser/assessors"
SCORES_DIR="/home/rothermm/THINGS/03_results/assessor_scores/subj${SUBJ_PAD}/800split"
PLOTS_DIR="/home/rothermm/THINGS/03_results/plots/800split"
mkdir -p "$SCORES_DIR" "$PLOTS_DIR"

# Conda env
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

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "/home/rothermm/THINGS/01_scripts/logs/10_vdvaeAssess800_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
