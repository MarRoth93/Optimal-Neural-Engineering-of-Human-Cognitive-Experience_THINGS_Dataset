#!/bin/bash
#SBATCH --job-name=plotShifted_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/plotShifted_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/plotShifted_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

SHIFTED_BASE="/home/rothermm/THINGS/03_results/vdvae_shifted/subj${SUBJ_PAD}"
PLOTS_DIR="/home/rothermm/THINGS/03_results/plots"

# Choose alphas to match what you decoded.
# For sigma-mode run:
ALPHAS=(-1 -0.5 0 0.5 1)
# For raw-mode run, you might prefer:
# ALPHAS=(-4 -3 -2 0 2 3 4)

IMG_SIZE=256
N_EX=4

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V

# Sanity
[[ -d "$SHIFTED_BASE" ]] || { echo "Missing: $SHIFTED_BASE" >&2; exit 1; }
mkdir -p "$PLOTS_DIR"

CMD=( python -u 16_plot_shifted_examples.py
  --sub "$SUBJ"
  --shifted_base "$SHIFTED_BASE"
  --plots_dir "$PLOTS_DIR"
  --n_examples "$N_EX"
  --tile "$IMG_SIZE"
  --alphas "${ALPHAS[@]}"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/plotShifted_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
