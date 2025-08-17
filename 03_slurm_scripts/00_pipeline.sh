#!/usr/bin/env bash
set -euo pipefail

# Load Conda
module purge
module load miniconda
echo "Loaded miniconda"

# Activate environment
source $CONDA_ROOT/bin/activate
eval "$(conda shell.bash hook)"
conda activate brain-diffuser
echo "Activated Conda environment: $(which python)"

# Directory that contains your sbatch scripts
DIR="/home/rothermm/THINGS/03_slurm_scripts"

# <<< Put the exact order you want here >>>
ORDER=(
  "01_make_vc_mask_voxelmeta_job.sh"
  "02_extract_betas_to_numpy_job.sh"
  "03_extract_h5_to_numpy_v2_job.sh"
  "04_split_and_average_things_job.sh"
  "04b_split_and_average_things_800_job.sh"
)

# Dependency mode: afterok (stop on failure) or afterany (continue even if a job fails)
DEP_MODE="${DEP_MODE:-afterok}"

# Safety: don't run if a dependency is invalid
KILL_ON_BAD="--kill-on-invalid-dep=yes"

declare -A JIDS
prev=""

echo "Submitting ${#ORDER[@]} jobs from $DIR (dependency: $DEP_MODE)"
for f in "${ORDER[@]}"; do
  path="$DIR/$f"
  [[ -f "$path" ]] || { echo "ERROR: Missing script $path" >&2; exit 1; }
  [[ -x "$path" ]] || chmod +x "$path"

  if [[ -n "$prev" ]]; then
    jid=$(sbatch --parsable --dependency="$DEP_MODE:$prev" $KILL_ON_BAD "$path")
  else
    jid=$(sbatch --parsable $KILL_ON_BAD "$path")
  fi

  echo "$f -> job $jid"
  JIDS["$f"]="$jid"
  prev="$jid"
done

# Write a quick manifest of the submission
out="$DIR/last_pipeline_submission.txt"
{
  date
  echo "Dependency mode: $DEP_MODE"
  for f in "${ORDER[@]}"; do
    printf "%-45s %s\n" "$f" "${JIDS[$f]}"
  done
} | tee "$out"
echo "Wrote manifest to: $out"
