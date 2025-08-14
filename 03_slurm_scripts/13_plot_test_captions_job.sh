#!/bin/bash
#SBATCH --job-name=plotCaptions_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/plotCaptions_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/plotCaptions_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)
TEST_PATHS="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/test_image_paths.txt"
CAP_JSONL="/home/rothermm/THINGS/02_data/captions/subj${SUBJ_PAD}/test_captions.jsonl"
OUT_PNG="/home/rothermm/THINGS/03_results/plots/caption_test.png"

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V

[[ -f "$TEST_PATHS" ]] || { echo "Missing: $TEST_PATHS" >&2; exit 1; }
[[ -f "$CAP_JSONL" ]] || { echo "Missing: $CAP_JSONL" >&2; exit 1; }
mkdir -p "$(dirname "$OUT_PNG")"

python -u 13_plot_test_captions.py \
  --sub "$SUBJ" \
  --test_paths "$TEST_PATHS" \
  --captions_jsonl "$CAP_JSONL" \
  --out_png "$OUT_PNG" \
  --n 5 | tee "/home/rothermm/THINGS/01_scripts/logs/plotCaptions_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
