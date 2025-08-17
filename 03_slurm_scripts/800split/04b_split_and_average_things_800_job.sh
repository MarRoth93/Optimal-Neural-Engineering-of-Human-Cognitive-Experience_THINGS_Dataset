#!/bin/bash
#SBATCH --job-name=thingsSplit800_s01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/04b_thingsSplit_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/04b_thingsSplit_s01_%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

echo "==== Job started on $(hostname) at $(date) ===="
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"
mkdir -p /home/rothermm/THINGS/01_scripts/logs

# -------- Paths --------
SCRIPTS_DIR="/home/rothermm/THINGS/01_scripts"

SPLIT_PY="${SCRIPTS_DIR}/04b_split_and_average_things_800.py"
MAIN_PY="${SCRIPTS_DIR}/04_split_and_average_things.py"
CHECK_PY="${SCRIPTS_DIR}/04c_check_split_and_outputs.py"  # optional

X_PATH="/home/rothermm/THINGS/02_data/preprocessed_data/subj01/X.npy"
STIM_TSV_IN="/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_stimulus-metadata.tsv"
OUTDIR="/home/rothermm/THINGS/02_data/preprocessed_data/subj01/800split"

IDCOL="stimulus"
N_TEST="800"
STRATEGY="first"
SEED="42"
AVGTRAIN=""

BASENAME="$(basename "$STIM_TSV_IN")"
STEM="${BASENAME%.*}"
STIM_TSV_OUT="$(dirname "$STIM_TSV_IN")/${STEM}_split${N_TEST}.tsv"

SPLIT_LOG="/home/rothermm/THINGS/01_scripts/logs/04c_check_split_${SLURM_JOB_ID}.split.log"
MAIN_LOG="/home/rothermm/THINGS/01_scripts/logs/04c_check_split_${SLURM_JOB_ID}.main.log"
CHECK_LOG="/home/rothermm/THINGS/01_scripts/logs/04c_check_split_${SLURM_JOB_ID}.check.log"

mkdir -p "$OUTDIR"

echo "Split helper : $SPLIT_PY"
echo "Main script  : $MAIN_PY"
echo "Stim IN      : $STIM_TSV_IN"
echo "Stim OUT     : $STIM_TSV_OUT"
echo "X.npy        : $X_PATH"
echo "Out dir      : $OUTDIR"
echo "ID column    : $IDCOL"
echo "N_TEST       : $N_TEST (strategy=$STRATEGY, seed=$SEED)"
echo "AVGTRAIN     : ${AVGTRAIN:-no}"

# ------------ STEP 1: Create split (pin original test + add extras) ------------
stdbuf -oL -eL python -u "$SPLIT_PY" \
  --stimulus_tsv "$STIM_TSV_IN" \
  --out_tsv      "$STIM_TSV_OUT" \
  --id_col       "$IDCOL" \
  --strategy     "$STRATEGY" \
  --seed         "$SEED" \
  --n_test       "$N_TEST" \
  | tee "$SPLIT_LOG"

# Ensure the split file exists and is non-empty
[[ -s "$STIM_TSV_OUT" ]] || { echo "ERROR: $STIM_TSV_OUT not created"; exit 1; }

# ------------ STEP 2: Split + average with the main script ---------------------
CMD2=( python -u "$MAIN_PY"
  --x_path       "$X_PATH"
  --stimulus_tsv "$STIM_TSV_OUT"
  --out_dir      "$OUTDIR"
  --id_col       "$IDCOL"
)
if [[ -n "$AVGTRAIN" ]]; then CMD2+=( --avg_train ); fi
printf 'Running (main): %q ' "${CMD2[@]}"; echo
stdbuf -oL -eL "${CMD2[@]}" | tee "$MAIN_LOG"

# Ensure expected outputs exist
[[ -s "$OUTDIR/X_test_avg.npy" ]] || { echo "ERROR: X_test_avg.npy missing in $OUTDIR"; exit 1; }
[[ -s "$OUTDIR/test_image_ids.txt" ]] || { echo "ERROR: test_image_ids.txt missing in $OUTDIR"; exit 1; }

# ------------ STEP 3: Optional sanity check -----------------------------------
if [[ -f "$CHECK_PY" ]]; then
  stdbuf -oL -eL python -u "$CHECK_PY" \
    --stimulus_tsv "$STIM_TSV_OUT" \
    --out_dir      "$OUTDIR" \
    --id_col       "$IDCOL" \
    --n_test       "$N_TEST" \
    --x_path       "$X_PATH" \
    | tee "$CHECK_LOG"
else
  echo "[WARN] $CHECK_PY not found; skipping checks."
fi

echo "==== Job finished at $(date) ===="
