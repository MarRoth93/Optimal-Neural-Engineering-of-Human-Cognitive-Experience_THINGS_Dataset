#!/bin/bash
#SBATCH --job-name=14_thetaFromScores_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/14_thetaFromScores_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/14_thetaFromScores_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

SCORES_DIR="/home/rothermm/THINGS/03_results/assessor_scores/subj${SUBJ_PAD}"
FEAT_NPZ="/home/rothermm/THINGS/02_data/extracted_features/subj${SUBJ_PAD}/things_vdvae_features_31l.npz"
OUT_DIR="/home/rothermm/THINGS/03_results/thetas/subj${SUBJ_PAD}"

# Conda env
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V

# Checks
[[ -f "${SCORES_DIR}/emonet_scores_sub${SUBJ_PAD}.pkl" ]] || { echo "Missing emonet scores" >&2; exit 1; }
[[ -f "${SCORES_DIR}/memnet_scores_sub${SUBJ_PAD}.pkl" ]] || { echo "Missing memnet scores" >&2; exit 1; }
[[ -f "$FEAT_NPZ" ]] || { echo "Missing features NPZ: $FEAT_NPZ" >&2; exit 1; }
mkdir -p "$OUT_DIR"

python -u 14_compute_theta_from_scores.py \
  --sub "$SUBJ" \
  --scores_dir "$SCORES_DIR" \
  --features_npz "$FEAT_NPZ" \
  --out_dir "$OUT_DIR" \
  --which decoded \
  --frac 0.10 \
  --normalize \
  | tee "/home/rothermm/THINGS/01_scripts/logs/14_thetaFromScores_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
