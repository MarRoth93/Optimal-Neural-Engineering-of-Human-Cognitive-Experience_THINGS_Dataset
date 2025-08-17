#!/bin/bash
#SBATCH --job-name=vc_mask_s01
#SBATCH --ntasks=1
#SBATCH --output=logs/01_vc_mask_s01_%j.out
#SBATCH --error=logs/01_vc_mask_s01_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/

# Debug info
echo "==== Job started on $(hostname) at $(date) ===="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# Load Conda
module purge
module load miniconda
echo "Loaded miniconda"

# Activate environment
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

# Paths for script arguments (if any are needed inside Python)
BIDS_ROOT="/home/rothermm/THINGS/02_data"
DERIV_ROOT="/home/rothermm/THINGS/02_data/derivatives"

# Run the mask creation script
echo "Running make_vc_mask_voxelmeta.py ..."
python -u 01_make_vc_mask_voxelmeta.py | tee logs/01_vc_mask_s01_${SLURM_JOB_ID}.debug.log

echo "==== Job finished at $(date) ===="
