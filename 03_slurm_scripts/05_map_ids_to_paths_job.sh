#!/bin/bash
#SBATCH --job-name=thingsMapPaths_s01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/05_mapPaths_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/05_mapPaths_s01_%j.err

set -euo pipefail

echo "==== Job started on $(hostname) at $(date) ===="

# NOTE: Slurm creates the log files before the script runs.
# Make sure /home/rothermm/THINGS/01_scripts/logs exists ahead of time, or keep this line and submit once the dir exists.
mkdir -p /home/rothermm/THINGS/01_scripts/logs

OUTDIR="/home/rothermm/THINGS/02_data/preprocessed_data/subj01"
mkdir -p "$OUTDIR"

# Conda env (same recipe that worked for you)
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"; python -V

IMGROOT="/home/rothermm/THINGS/02_data/stimuli/images"
test -d "$IMGROOT" || { echo "Images root not found: $IMGROOT" >&2; exit 1; }
test -f "$OUTDIR/train_image_ids.txt" || { echo "Missing: $OUTDIR/train_image_ids.txt" >&2; exit 1; }
test -f "$OUTDIR/test_image_ids.txt"  || { echo "Missing: $OUTDIR/test_image_ids.txt"  >&2; exit 1; }

echo "Using IMGROOT=$IMGROOT"
echo "OUTDIR=$OUTDIR"

# Toggle this to 1 to auto-pick when duplicates exist
ALLOW_DUPLICATES=0
EXTRA_OPTS=()
if [[ "$ALLOW_DUPLICATES" -eq 1 ]]; then
  EXTRA_OPTS+=(--allow-duplicates)
fi

# Train
srun -u python 05_map_ids_to_paths.py \
  --ids_file "$OUTDIR/train_image_ids.txt" \
  --images_root "$IMGROOT" \
  --out_file "$OUTDIR/train_image_paths.txt" \
  --debug "${EXTRA_OPTS[@]}"

# Test
srun -u python 05_map_ids_to_paths.py \
  --ids_file "$OUTDIR/test_image_ids.txt" \
  --images_root "$IMGROOT" \
  --out_file "$OUTDIR/test_image_paths.txt" \
  --debug "${EXTRA_OPTS[@]}"

# Sanity check: counts should match
for SPLIT in train test; do
  ids_n=$(wc -l < "$OUTDIR/${SPLIT}_image_ids.txt" || echo 0)
  paths_n=$(wc -l < "$OUTDIR/${SPLIT}_image_paths.txt" || echo 0)
  echo "$SPLIT: ids=$ids_n  paths=$paths_n"
  if [ "$ids_n" -ne "$paths_n" ]; then
    echo "WARNING: $SPLIT mismatch (ids vs. paths)" >&2
  fi
done

echo "==== Job finished at $(date) ===="
