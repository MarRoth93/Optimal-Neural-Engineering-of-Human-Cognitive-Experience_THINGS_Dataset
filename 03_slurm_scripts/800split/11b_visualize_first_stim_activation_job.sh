#!/bin/bash
#SBATCH --job-name=vizFirstStim800_s01
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/11_vizFirstStim800_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/11_vizFirstStim800_s01_%j.err

set -euo pipefail
echo "==== Job started on $(hostname) at $(date) ===="
mkdir -p /home/rothermm/THINGS/01_scripts/logs

SUBJ=1
SUBJ_PAD=$(printf "%02d" $SUBJ)

BRAINMASK="/home/rothermm/THINGS/02_data/derivatives/fmriprep/sub-01/anat/sub-01_space-T1w_mask.nii.gz"
VOXEL_TSV="/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_voxel-metadata.tsv"
PREPROC_DIR="/home/rothermm/THINGS/02_data/preprocessed_data/subj${SUBJ_PAD}/800split"
OUT_DIR="/home/rothermm/THINGS/03_results/visualizations/subj${SUBJ_PAD}/800split"
mkdir -p "$OUT_DIR"

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V

for f in "$BRAINMASK" "$VOXEL_TSV"; do
  [[ -e "$f" ]] || { echo "Missing required file: $f" >&2; exit 1; }
done
[[ -d "$PREPROC_DIR" ]] || { echo "Missing dir: $PREPROC_DIR" >&2; exit 1; }

CMD=( python -u 11_visualize_first_stim_activation.py
  --sub "$SUBJ"
  --brainmask "$BRAINMASK"
  --voxel_tsv "$VOXEL_TSV"
  --preproc_dir "$PREPROC_DIR"
  --out_dir "$OUT_DIR"
)

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "/home/rothermm/THINGS/01_scripts/logs/11_vizFirstStim800_s01_${SLURM_JOB_ID}.debug.log"
echo "==== Job finished at $(date) ===="
