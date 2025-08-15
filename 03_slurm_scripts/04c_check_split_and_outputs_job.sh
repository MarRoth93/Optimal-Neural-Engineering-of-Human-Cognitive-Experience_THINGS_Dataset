#!/bin/bash
#SBATCH --job-name=thingsCheck_s01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/04c_thingsCheck_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/04c_thingsCheck_s01_%j.err

echo "==== Job started on $(hostname) at $(date) ===="
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"
mkdir -p /home/rothermm/THINGS/01_scripts/logs

# ---- Paths & knobs ----
SCRIPTS_DIR="/home/rothermm/THINGS/01_scripts"
CHECK_PY="${SCRIPTS_DIR}/04c_check_split_and_outputs.py"

STIM_TSV="/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_stimulus-metadata_split800.tsv"
OUTDIR="/home/rothermm/THINGS/02_data/preprocessed_data/subj01"
X_PATH="/home/rothermm/THINGS/02_data/preprocessed_data/subj01/X.npy"  # optional but recommended
IDCOL="stimulus"
N_TEST="800"

CMD=( python -u "$CHECK_PY"
  --stimulus_tsv "$STIM_TSV"
  --out_dir      "$OUTDIR"
  --id_col       "$IDCOL"
  --n_test       "$N_TEST"
  --x_path       "$X_PATH"
)

echo "Running (check): ${CMD[*]}"
"${CMD[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/thingsCheck_s01_${SLURM_JOB_ID}.log"

echo "==== Job finished at $(date) ===="
