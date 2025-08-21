#!/bin/bash
#SBATCH --job-name=19_ssim_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/19_ssim_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/19_ssim_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="

mkdir -p /home/rothermm/THINGS/01_scripts/logs
mkdir -p /home/rothermm/THINGS/03_results/plots/ssim

SUBJ=1
SUBJ_PAD=$(printf "%02d" "$SUBJ")

# paths from your description
SHIFTED_BASE="/home/rothermm/THINGS/03_results/vdvae_shifted"
STIM_PATHS_ROOT="/home/rothermm/THINGS/02_data/preprocessed_data"
PLOTS_DIR="/home/rothermm/THINGS/03_results/plots"

# Alphas present in your tree
ALPHAS=(-4 -3 -2 -1 -0.5 0 0.5 1 2 3 4)

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V

CMD=( python -u 19_ssim.py
  --sub "$SUBJ"
  --shifted_base "$SHIFTED_BASE"
  --stim_paths_root "$STIM_PATHS_ROOT"
  --plots_dir "$PLOTS_DIR"
  --models emonet memnet
  --alphas "${ALPHAS[@]}"
  --min_n 5
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/19_ssim_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
