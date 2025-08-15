#!/bin/bash
#SBATCH --job-name=vdvaeShift_s01
#SBATCH --partition=gpu
#SBATCH --gpus=a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/15_vdvaeShift_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/15_vdvaeShift_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

# *** Force the ridge latents (the same you used in 08_decode_vdvae_from_pred.py) ***
PRED_NPY="/home/rothermm/THINGS/02_data/predicted_features/subj${SUBJ_PAD}/things_vdvae_pred_sub${SUBJ_PAD}_31l_alpha50000.npy"

FEAT_DIR="/home/rothermm/THINGS/02_data/extracted_features/subj${SUBJ_PAD}"
THETA_EMO="/home/rothermm/THINGS/03_results/thetas/subj${SUBJ_PAD}/theta_emonet_decoded_top10_minus_bottom10.npy"
THETA_MEM="/home/rothermm/THINGS/03_results/thetas/subj${SUBJ_PAD}/theta_memnet_decoded_top10_minus_bottom10.npy"
OUT_BASE="/home/rothermm/THINGS/03_results/vdvae_shifted/subj${SUBJ_PAD}"
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

# Start with small, safe steps (1 std along theta direction)
ALPHAS=(-1 -0.5 0 0.5 1)

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
