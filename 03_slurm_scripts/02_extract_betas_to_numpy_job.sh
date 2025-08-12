#!/bin/bash
#SBATCH --job-name=thingsX_s01
#SBATCH --ntasks=1
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/thingsX_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/thingsX_s01_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/

echo "==== Job started on $(hostname) at $(date) ===="
module purge
module load miniconda
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"

MASK="/home/rothermm/THINGS/01_scripts/sub-01_visualcortex_mask.nii.gz"
BETAS_GLOB="/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/**/**/*.nii.gz"   # <<< EDIT ME
OUT_DIR="/home/rothermm/THINGS/01_scripts/out_sub-01_X"                                   # or your brain-diffuser data dir
EVENTS_ROOT="/home/rothermm/THINGS/02_data"

echo "Running extraction..."
python -u 02_extract_betas_to_numpy.py \
  --mask "$MASK" \
  --betas_glob "$BETAS_GLOB" \
  --out_dir "$OUT_DIR" \
  --events_root "$EVENTS_ROOT" \
  | tee /home/rothermm/THINGS/01_scripts/logs/thingsX_s01_${SLURM_JOB_ID}.debug.log

echo "==== Job finished at $(date) ===="
