#!/usr/bin/env bash
#SBATCH --job-name=14c_theta-regression
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/%x_sub%a_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/%x_sub%a_%j.err
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=1

set -euo pipefail
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser

SUBJ=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")

python /home/rothermm/THINGS/01_scripts/14c_compute_theta_regression.py \
  --sub "${SUBJ}" \
  --features_npz /home/rothermm/THINGS/02_data/extracted_features/subj${SUBJ}/things_vdvae_features_31l.npz \
  --scores_dir /home/rothermm/THINGS/03_results/assessor_scores/subj${SUBJ} \
  --assessors emonet memnet \
  --out_dir /home/rothermm/THINGS/03_results/thetas_regression/subj${SUBJ}
