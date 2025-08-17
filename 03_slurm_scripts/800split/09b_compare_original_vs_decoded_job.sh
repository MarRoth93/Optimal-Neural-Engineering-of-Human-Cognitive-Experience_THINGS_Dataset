#!/bin/bash
#SBATCH --job-name=vdvaeCompare800_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/09_vdvaeCompare800_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/09_vdvaeCompare800_s01_%j.err

set -euo pipefail

echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)
TEST_PATHS="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/800split/test_image_paths.txt"
DECODED_DIR="/home/rothermm/THINGS/03_results/vdvae/subj${SUBJ_PAD}/800split"
OUT_PNG="/home/rothermm/THINGS/03_results/plots/800split/compare_first10_sub${SUBJ_PAD}.png"
N=10
IMG_SIZE=256

# ensure plot folder exists
mkdir -p "$(dirname "$OUT_PNG")"

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

CMD=( python -u 09_compare_original_vs_decoded.py
  --sub "$SUBJ"
  --test_paths "$TEST_PATHS"
  --decoded_dir "$DECODED_DIR"
  --out_png "$OUT_PNG"
  --n "$N"
  --img_size "$IMG_SIZE"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "/home/rothermm/THINGS/01_scripts/logs/09_vdvaeCompare800_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
