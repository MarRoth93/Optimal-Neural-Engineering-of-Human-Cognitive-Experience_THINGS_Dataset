#!/bin/bash
#SBATCH --job-name=thingsMapPaths_s01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --chdir=/home/rothermm/THINGS/01_scripts/
#SBATCH --output=/home/rothermm/THINGS/01_scripts/logs/mapPaths_s01_%j.out
#SBATCH --error=/home/rothermm/THINGS/01_scripts/logs/mapPaths_s01_%j.err

echo "==== Job started on $(hostname) at $(date) ===="
module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Python: $(which python)"

IMGROOT="/home/rothermm/THINGS/02_data/stimuli/images"
OUTDIR="/home/rothermm/THINGS/02_data/preprocessed_data/subj01"

python -u 05_map_ids_to_paths.py \
  --ids_file "$OUTDIR/train_image_ids.txt" \
  --images_root "$IMGROOT" \
  --out_file "$OUTDIR/train_image_paths.txt"

python -u 05_map_ids_to_paths.py \
  --ids_file "$OUTDIR/test_image_ids.txt" \
  --images_root "$IMGROOT" \
  --out_file "$OUTDIR/test_image_paths.txt"

echo "==== Job finished at $(date) ===="
