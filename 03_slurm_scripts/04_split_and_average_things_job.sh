#!/bin/bash
#SBATCH --job-name=04_Split
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/04_thingsSplit_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/04_thingsSplit_s01_%j.err

echo "==== Job started on $(hostname) at $(date) ===="
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"
mkdir -p /home/rothermm/THINGS/01_scripts/logs

X_PATH="/home/rothermm/THINGS/02_data/preprocessed_data/subj01/X.npy"
STIM_TSV="/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_stimulus-metadata.tsv"
OUTDIR="/home/rothermm/THINGS/02_data/preprocessed_data/subj01"
IDCOL="stimulus"
AVGTRAIN=""   # set to "1" to average train as well

CMD=( python -u 04_split_and_average_things.py
  --x_path "$X_PATH"
  --stimulus_tsv "$STIM_TSV"
  --out_dir "$OUTDIR"
  --id_col "$IDCOL"
)

if [[ -n "$AVGTRAIN" ]]; then
  CMD+=( --avg_train )
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/04_thingsSplit_s01_${SLURM_JOB_ID}.debug.log"

echo "==== Job finished at $(date) ===="
