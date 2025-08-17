#!/bin/bash
#SBATCH --job-name=vdvaeRidge_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/07_vdvaeRidge_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/07_vdvaeRidge_s01_%j.err

set -euo pipefail

echo "==== Job started on $(hostname) at $(date) ===="

# Logs dir
mkdir -p /home/rothermm/THINGS/01_scripts/logs

# Use all allocated CPU threads in NumPy/MKL/BLAS
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- config you can tweak ---
SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)
ALPHA=60000

FEATURES_NPZ="/home/rothermm/THINGS/02_data/extracted_features/subj${SUBJ_PAD}/things_vdvae_features_31l.npz"
TRAIN_FMRI="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/X_train.npy"
TEST_FMRI="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/X_test_avg.npy"
OUT_PRED_DIR="/home/rothermm/THINGS/02_data/predicted_features/subj${SUBJ_PAD}"
OUT_WEIGHTS_DIR="/home/rothermm/THINGS/02_data/regression_weights/subj${SUBJ_PAD}"
# ----------------------------

# Conda env (same recipe that worked for you)
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"
python -V

# Sanity checks
for f in "$FEATURES_NPZ" "$TRAIN_FMRI" "$TEST_FMRI"; do
  [[ -e "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done
mkdir -p "$OUT_PRED_DIR" "$OUT_WEIGHTS_DIR"

# Run regression
CMD=( python -u 07_regress_brain_to_vdvae.py
  --sub "$SUBJ"
  --alpha "$ALPHA"
  --features_npz "$FEATURES_NPZ"
  --train_fmri "$TRAIN_FMRI"
  --test_fmri  "$TEST_FMRI"
  --out_pred_dir "$OUT_PRED_DIR"
  --out_weights_dir "$OUT_WEIGHTS_DIR"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/07_vdvaeRidge_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
