#!/bin/bash
#SBATCH --job-name=vdvaeShiftDecode_s01
#SBATCH --partition=gpu
#SBATCH --gpus=a100_80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/vdvaeShiftDecode_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/vdvaeShiftDecode_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

VDVAE_ROOT="/home/rothermm/brain-diffuser/vdvae"
MODEL_DIR="/home/rothermm/brain-diffuser/vdvae/model"
FEAT_DIR="/home/rothermm/THINGS/02_data/extracted_features/subj${SUBJ_PAD}"
OUT_BASE="/home/rothermm/THINGS/03_results/vdvae_shifted/subj${SUBJ_PAD}"

# Prefer DNN predictions, fallback to ridge
PRED_DNN="/home/rothermm/THINGS/02_data/predicted_features/subj${SUBJ_PAD}/things_vdvae_pred_sub${SUBJ_PAD}_31l_DNN_w2048_d6.npy"
PRED_RIDGE="/home/rothermm/THINGS/02_data/predicted_features/subj${SUBJ_PAD}/things_vdvae_pred_sub${SUBJ_PAD}_31l_alpha50000.npy"
if [[ -f "$PRED_DNN" ]]; then
  PRED_NPY="$PRED_DNN"
elif [[ -f "$PRED_RIDGE" ]]; then
  PRED_NPY="$PRED_RIDGE"
else
  echo "Missing predicted latents (.npy). Tried:"
  echo "  $PRED_DNN"
  echo "  $PRED_RIDGE"
  exit 1
fi

THETA_EMO="/home/rothermm/THINGS/03_results/thetas/subj${SUBJ_PAD}/theta_emonet_decoded_top10_minus_bottom10.npy"
THETA_MEM="/home/rothermm/THINGS/03_results/thetas/subj${SUBJ_PAD}/theta_memnet_decoded_top10_minus_bottom10.npy"

# Env
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V; nvidia-smi || true

# Checks
[[ -f "${FEAT_DIR}/ref_latents.npz" ]] || { echo "Missing ref_latents.npz" >&2; exit 1; }
[[ -f "$THETA_EMO" ]] || { echo "Missing theta_emo: $THETA_EMO" >&2; exit 1; }
[[ -f "$THETA_MEM" ]] || { echo "Missing theta_mem: $THETA_MEM" >&2; exit 1; }
mkdir -p "$OUT_BASE"

CMD=( python -u 15_shift_and_decode_vdvae.py
  --sub "$SUBJ"
  --pred_npy "$PRED_NPY"
  --feat_dir "$FEAT_DIR"
  --theta_emo "$THETA_EMO"
  --theta_mem "$THETA_MEM"
  --which both
  --alphas -4 -3 -2 0 2 3 4
  --bs 30
  --vdvae_root "$VDVAE_ROOT"
  --model_dir  "$MODEL_DIR"
  --out_base   "$OUT_BASE"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/vdvaeShiftDecode_s01_${SLURM_JOB_ID}.debug.log"
echo "==== Job finished at $(date) ===="
