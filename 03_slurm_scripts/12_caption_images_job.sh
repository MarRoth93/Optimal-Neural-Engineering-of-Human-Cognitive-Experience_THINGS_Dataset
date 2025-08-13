#!/bin/bash
#SBATCH --job-name=captionTHINGS_s01
#SBATCH --partition=gpu
#SBATCH --gpus=a100_80gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/captionTHINGS_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/captionTHINGS_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

TRAIN_PATHS="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/train_image_paths.txt"
TEST_PATHS="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/test_image_paths.txt"
OUT_DIR="/home/rothermm/THINGS/02_data/captions/subj${SUBJ_PAD}"


MODEL="Salesforce/blip2-opt-2.7b"
BATCH=8
MAXTOK=30
PROMPT=""   # e.g., "A high-quality photo of"

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V; nvidia-smi || true

[[ -f "$TRAIN_PATHS" ]] || { echo "Missing: $TRAIN_PATHS" >&2; exit 1; }
[[ -f "$TEST_PATHS"  ]] || { echo "Missing: $TEST_PATHS"  >&2; exit 1; }
mkdir -p "$OUT_DIR"

CMD=( python -u 12_caption_images.py
  --sub "$SUBJ"
  --train_paths "$TRAIN_PATHS"
  --test_paths  "$TEST_PATHS"
  --out_dir     "$OUT_DIR"
  --model "$MODEL"
  --batch_size "$BATCH"
  --max_new_tokens "$MAXTOK"
  --image_size 384
)

# optional prompt / beam search
if [[ -n "$PROMPT" ]]; then
  CMD+=( --prompt "$PROMPT" )
fi
# enable beam search for better captions (slower)
CMD+=( --beam_search --num_beams 5 )

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "/home/rothermm/THINGS/01_scripts/logs/captionTHINGS_s01_${SLURM_JOB_ID}.debug.log"
echo "==== Job finished at $(date) ===="
