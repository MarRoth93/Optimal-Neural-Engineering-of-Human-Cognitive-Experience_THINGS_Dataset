#!/usr/bin/env bash
#SBATCH --job-name=15b_theta-eval
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/%x_sub%a_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/%x_sub%a_%j.err
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=1

set -euo pipefail
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser

SUBJ=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")

python /home/rothermm/THINGS/01_scripts/15b_evaluate_theta_sweep.py \
  --sub "${SUBJ}"
