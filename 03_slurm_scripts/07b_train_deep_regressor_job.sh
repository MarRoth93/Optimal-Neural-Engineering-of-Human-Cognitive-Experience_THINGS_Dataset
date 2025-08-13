#!/bin/bash
#SBATCH --job-name=dnnRidgefree_s01
#SBATCH --partition=gpu              # <- use the GPU queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # good default for data loading / BLAS
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/dnnRidgefree_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/dnnRidgefree_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

# Prevent BLAS oversubscription
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

FEATURES_NPZ="/home/rothermm/THINGS/02_data/extracted_features/subj${SUBJ_PAD}/things_vdvae_features_31l.npz"
TRAIN_FMRI="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/X_train.npy"
TEST_FMRI="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/X_test_avg.npy"
OUT_PRED_DIR="/home/rothermm/THINGS/02_data/predicted_features/subj${SUBJ_PAD}"
OUT_CKPT_DIR="/home/rothermm/THINGS/02_data/learned_models/subj${SUBJ_PAD}"

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V; nvidia-smi || true

for f in "$FEATURES_NPZ" "$TRAIN_FMRI" "$TEST_FMRI"; do
  [[ -e "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done
mkdir -p "$OUT_PRED_DIR" "$OUT_CKPT_DIR"

CMD=( python -u 07b_train_deep_regressor.py
  --sub "$SUBJ"
  --features_npz "$FEATURES_NPZ"
  --train_fmri   "$TRAIN_FMRI"
  --test_fmri    "$TEST_FMRI"
  --out_pred_dir "$OUT_PRED_DIR"
  --out_ckpt_dir "$OUT_CKPT_DIR"
  --width 1024 --depth 6 --drop 0.3
  --batch_size 256 --epochs 80 --lr 3e-4 --wd 1e-3
  --val_frac 0.1 --early_stop 5
  --w_cos 1.0 --w_mse 1.0
  --keep_voxels 10000
  --input_noise 0.02
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "/home/rothermm/THINGS/01_scripts/logs/dnnRidgefree_s01_${SLURM_JOB_ID}.debug.log"
echo "==== Job finished at $(date) ===="
