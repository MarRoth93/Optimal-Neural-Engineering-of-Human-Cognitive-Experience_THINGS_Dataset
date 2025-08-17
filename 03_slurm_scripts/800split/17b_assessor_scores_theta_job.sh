#!/bin/bash
#SBATCH --job-name=vdvaeAssessTheta800_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/17_vdvaeAssessTheta800_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/17_vdvaeAssessTheta800_s01_%j.err

set -euo pipefail

echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

TEST_PATHS="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/800split/test_image_paths.txt"
SHIFTED_ROOT="/home/rothermm/THINGS/03_results/vdvae_shifted/subj${SUBJ_PAD}/800split"
ASSESSORS_ROOT="/home/rothermm/brain-diffuser/assessors"
SCORES_DIR="/home/rothermm/THINGS/03_results/assessor_scores/subj${SUBJ_PAD}/800split/theta"
PLOTS_DIR="/home/rothermm/THINGS/03_results/plots/800split/theta"
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
[[ -d "$SHIFTED_ROOT" ]] || { echo "Missing shifted root: $SHIFTED_ROOT" >&2; exit 1; }
[[ -e "${ASSESSORS_ROOT}/image_mean.npy" ]] || { echo "Missing MemNet mean: ${ASSESSORS_ROOT}/image_mean.npy" >&2; exit 1; }

# Run scoring & plotting for all discovered alphas (EmoNet + MemNet)
CMD=( python -u 17_assessor_scores_theta.py
  --sub "$SUBJ"
  --test_paths "$TEST_PATHS"
  --shifted_root "$SHIFTED_ROOT"
  --assessors_root "$ASSESSORS_ROOT"
  --out_dir "$SCORES_DIR"
  --plots_dir "$PLOTS_DIR"
  --device auto
  --batch_size 96
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "/home/rothermm/THINGS/01_scripts/logs/17_vdvaeAssessTheta800_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
