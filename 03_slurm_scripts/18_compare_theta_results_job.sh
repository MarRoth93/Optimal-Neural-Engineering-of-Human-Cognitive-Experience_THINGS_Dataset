#!/usr/bin/env bash
#SBATCH --job-name=18_compare_theta_results
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/%x_sub%a_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/%x_sub%a_%j.err
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=1    # subjects 1..8; adjust as needed

set -euo pipefail

# --- Conda env ---
module purge
module load miniconda
source "${CONDA_ROOT}/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser

echo "[INFO] Python: $(which python)"

SUBJ="${SLURM_ARRAY_TASK_ID}"

REPO_ROOT="/home/rothermm/THINGS"
SCRIPT="${REPO_ROOT}/01_scripts/18_compare_theta_results.py"
PLOT_DIR="/home/rothermm/THINGS/03_results/plots/theta"

# Ensure log & plot dirs exist
mkdir -p "/home/rothermm/THINGS/03_logs" "${PLOT_DIR}"

python "${SCRIPT}" \
  --sub "${SUBJ}" \
  --assessors both \
  --out_dir "${PLOT_DIR}"

echo "[DONE] subject ${SUBJ}"
