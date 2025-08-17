#!/bin/bash
#SBATCH --job-name=vdvaeDecode_s01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/08_vdvaeDecode_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/08_vdvaeDecode_s01_%j.err

set -euo pipefail

echo "==== Job started on $(hostname) at $(date) ===="

mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)
ALPHA=50000
BS=30

VDVAE_ROOT="/home/rothermm/brain-diffuser/vdvae"
MODEL_DIR="/home/rothermm/brain-diffuser/vdvae/model"
PRED_DIR="/home/rothermm/THINGS/02_data/predicted_features/subj${SUBJ_PAD}"
FEAT_DIR="/home/rothermm/THINGS/02_data/extracted_features/subj${SUBJ_PAD}"
OUT_DIR="/home/rothermm/THINGS/03_results/vdvae/subj${SUBJ_PAD}"
mkdir -p "$OUT_DIR"

# Conda env (your working method)
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V; nvidia-smi || true

# Sanity checks
ALPHA_TAG=$(python - <<PY
a=$ALPHA
print(str(int(a)) if float(a).is_integer() else str(a).replace('.','p'))
PY
)
PRED_NPY="${PRED_DIR}/things_vdvae_pred_sub${SUBJ_PAD}_31l_alpha${ALPHA_TAG}.npy"
[[ -e "$PRED_NPY" ]] || { echo "Missing predicted latents: $PRED_NPY" >&2; exit 1; }
[[ -e "${FEAT_DIR}/ref_latents.npz" ]] || { echo "Missing ref_latents: ${FEAT_DIR}/ref_latents.npz" >&2; exit 1; }

CMD=( python -u 08_decode_vdvae_from_pred.py
  --sub "$SUBJ"
  --alpha "$ALPHA"
  --bs "$BS"
  --vdvae_root "$VDVAE_ROOT"
  --model_dir  "$MODEL_DIR"
  --pred_dir   "$PRED_DIR"
  --feat_dir   "$FEAT_DIR"
  --out_dir    "$OUT_DIR"
)

# change the run line to merge stderr into stdout before tee
echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "/home/rothermm/THINGS/01_scripts/logs/08_vdvaeDecode_s01_${SLURM_JOB_ID}.debug.log"


echo "==== Job finished at $(date) ===="
