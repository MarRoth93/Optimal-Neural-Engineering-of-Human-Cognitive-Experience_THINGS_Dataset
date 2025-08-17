#!/bin/bash
#SBATCH --job-name=vdvaeShift800_s01
#SBATCH --partition=gpu
#SBATCH --gpus=a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/15_vdvaeShift800_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/15_vdvaeShift800_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

# 800split inputs
PRED_NPY="/home/rothermm/THINGS/02_data/predicted_features/subj${SUBJ_PAD}/800split/things_vdvae_pred_sub${SUBJ_PAD}_31l_alpha50000.npy"
FEAT_DIR="/home/rothermm/THINGS/02_data/extracted_features/subj${SUBJ_PAD}/800split"
THETA_EMO="/home/rothermm/THINGS/03_results/thetas/subj${SUBJ_PAD}/800split/theta_emonet_decoded_top10_minus_bottom10.npy"
THETA_MEM="/home/rothermm/THINGS/03_results/thetas/subj${SUBJ_PAD}/800split/theta_memnet_decoded_top10_minus_bottom10.npy"
OUT_BASE="/home/rothermm/THINGS/03_results/vdvae_shifted/subj${SUBJ_PAD}/800split"

VDVAE_ROOT="/home/rothermm/brain-diffuser/vdvae"
MODEL_DIR="/home/rothermm/brain-diffuser/vdvae/model"

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V; nvidia-smi || true

for f in "$PRED_NPY" "$THETA_EMO" "$THETA_MEM" "$FEAT_DIR/ref_latents.npz"; do
  [[ -e "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done
mkdir -p "$OUT_BASE"

# Shift magnitudes (in σ along theta direction)
ALPHAS=(4 3 -2 -1 -0.5 0 0.5 1 2 3 4)

python -u 15_shift_and_decode_vdvae.py \
  --sub "$SUBJ" \
  --pred_npy "$PRED_NPY" \
  --feat_dir "$FEAT_DIR" \
  --theta_emo "$THETA_EMO" \
  --theta_mem "$THETA_MEM" \
  --which both \
  --alphas "${ALPHAS[@]}" \
  --alpha_mode sigma \
  --bs 30 \
  --vdvae_root "$VDVAE_ROOT" \
  --model_dir "$MODEL_DIR" \
  --out_base "$OUT_BASE"

echo "==== Job finished at $(date) ===="
