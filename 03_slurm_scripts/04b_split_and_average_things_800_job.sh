#!/bin/bash
#SBATCH --job-name=thingsSplit_s01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/04b_thingsSplit_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/04b_thingsSplit_s01_%j.err

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

# (1) Split helper (your 04b is a split script that expects --out_tsv, etc.)
SPLIT_PY="${SCRIPTS_DIR}/04b_split_and_average_things_800.py"

# (2) Main split+average script (the one you posted first)
MAIN_PY="${SCRIPTS_DIR}/04_split_and_average_things.py"

# (3) Checker
CHECK_PY="${SCRIPTS_DIR}/04c_check_split_and_outputs.py"

# -------- Subject-specific IO --------
X_PATH="/home/rothermm/THINGS/02_data/preprocessed_data/subj01/X.npy"
STIM_TSV_IN="/home/rothermm/THINGS/02_data/derivatives/ICA-betas/sub-01/voxel-metadata/sub-01_task-things_stimulus-metadata.tsv"
OUTDIR="/home/rothermm/THINGS/02_data/preprocessed_data/subj01"

# -------- Split config --------
IDCOL="stimulus"
N_TEST="800"
STRATEGY="first"   # or "random"
SEED="42"
AVGTRAIN=""        # set to "1" to also average train in MAIN_PY

# Derived output metadata path (same folder as input, with suffix)
BASENAME="$(basename "$STIM_TSV_IN")"                     # sub-01_task-things_stimulus-metadata.tsv
STEM="${BASENAME%.*}"
STIM_TSV_OUT="$(dirname "$STIM_TSV_IN")/${STEM}_split${N_TEST}.tsv"

echo "Split helper : $SPLIT_PY"
echo "Main script  : $MAIN_PY"
echo "Checker      : $CHECK_PY"
echo "Stim IN      : $STIM_TSV_IN"
echo "Stim OUT     : $STIM_TSV_OUT"
echo "X.npy        : $X_PATH"
echo "Out dir      : $OUTDIR"
echo "ID column    : $IDCOL"
echo "N_TEST       : $N_TEST (strategy=$STRATEGY, seed=$SEED)"
echo "AVGTRAIN     : ${AVGTRAIN:-no}"

# ------------ STEP 1: Create split (exactly 800 unique test images) ------------
CMD1=( python -u "$SPLIT_PY"
  --stimulus_tsv "$STIM_TSV_IN"
  --out_tsv      "$STIM_TSV_OUT"
  --id_col       "$IDCOL"
  --strategy     "$STRATEGY"
  --seed         "$SEED"
  --n_test       "$N_TEST"
)
echo "Running (split): ${CMD1[*]}"
"${CMD1[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/thingsSplit_s01_${SLURM_JOB_ID}.split.log"

# ------------ STEP 2: Split + average with the main script ---------------------
mkdir -p "$OUTDIR"
CMD2=( python -u "$MAIN_PY"
  --x_path       "$X_PATH"
  --stimulus_tsv "$STIM_TSV_OUT"
  --out_dir      "$OUTDIR"
  --id_col       "$IDCOL"
)
if [[ -n "$AVGTRAIN" ]]; then
  CMD2+=( --avg_train )
fi
echo "Running (main): ${CMD2[*]}"
"${CMD2[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/thingsSplit_s01_${SLURM_JOB_ID}.main.log"

# ------------ STEP 3: Sanity check --------------------------------------------
CMD3=( python -u "$CHECK_PY"
  --stimulus_tsv "$STIM_TSV_OUT"
  --out_dir      "$OUTDIR"
  --id_col       "$IDCOL"
  --n_test       "$N_TEST"
  --x_path       "$X_PATH"
)
echo "Running (check): ${CMD3[*]}"
"${CMD3[@]}" | tee "/home/rothermm/THINGS/01_scripts/logs/thingsSplit_s01_${SLURM_JOB_ID}.check.log"

echo "==== Job finished at $(date) ===="
