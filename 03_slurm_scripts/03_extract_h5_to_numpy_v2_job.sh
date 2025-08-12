#!/bin/bash
#SBATCH --job-name=thingsH5_s01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/thingsH5_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/thingsH5_s01_%j.err

# ===== Debug header =====
echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"

# ===== Env =====
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"
mkdir -p /home/rothermm/THINGS/01_scripts/logs

# ===== Inputs (edit if your paths change) =====
H5="/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_voxel-wise-responses.h5"
VOXTSV="/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_voxel-metadata.tsv"
STIMTSV="/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_stimulus-metadata.tsv"
OUTDIR="/home/rothermm/THINGS/02_data/preprocessed_data/subj01"
ZSCORE_FRAC=""     # e.g., set to "0.8" to fit z-score on first 80% of trials

mkdir -p "$OUTDIR"

# ===== Run =====
CMD=( python -u 03_extract_h5_to_numpy_v2.py
  --h5 "$H5"
  --voxel_tsv "$VOXTSV"
  --stimulus_tsv "$STIMTSV"
  --out_dir "$OUTDIR"
)

# Optional z-score fraction
if [[ -n "$ZSCORE_FRAC" ]]; then
  CMD+=( --zscore_train_frac "$ZSCORE_FRAC" )
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/thingsH5_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
