#!/usr/bin/env bash
set -euo pipefail

module purge
module load miniconda
source "$CONDA_ROOT/bin/activate"
eval "$(conda shell.bash hook)"
conda activate brain-diffuser

DIR="/home/rothermm/THINGS/03_slurm_scripts"

ORDER=(
  # 100-split prep steps in parent dir
  "01_make_vc_mask_voxelmeta_job.sh"
  "02_extract_betas_to_numpy_job.sh"
  "03_extract_h5_to_numpy_v2_job.sh"

  # 800-split jobs live in subfolder; use exact filenames from your ls
  "800split/04b_split_and_average_things_800_job.sh"
  "800split/05b_map_ids_to_paths_job.sh"
  "800split/06b_extract_vdvae_features_job.sh"
  "800split/07b_regress_brain_to_vdvae_job.sh"
  "800split/08b_decode_vdvae_from_pred_job.sh"
  "800split/09b_compare_original_vs_decoded_job.sh"
  "800split/10b_compute_assessor_scores_job.sh"
  "800split/11b_visualize_first_stim_activation_job.sh"
  "800split/14b_compute_theta_from_scores_job.sh"
  "800split/15b_shift_and_decode_vdvae_job.sh"
  "800split/16b_plot_shifted_examples_job.sh"
  "800split/17b_assessor_scores_theta_job.sh"
)

DEP_MODE="${DEP_MODE:-afterok}"
KILL_ON_BAD="--kill-on-invalid-dep=yes"

# Preflight: show any missing entries up-front
missing=0
for f in "${ORDER[@]}"; do
  if [[ ! -f "$DIR/$f" ]]; then
    echo "MISSING: $DIR/$f" >&2
    missing=1
  fi
done
if [[ $missing -ne 0 ]]; then
  echo "Aborting due to missing scripts above." >&2
  exit 1
fi

declare -A JIDS
prev=""
echo "Submitting ${#ORDER[@]} jobs from $DIR (dependency: $DEP_MODE)"
for f in "${ORDER[@]}"; do
  path="$DIR/$f"
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

out="$DIR/last_pipeline_submission.txt"
{
  date
  echo "Dependency mode: $DEP_MODE"
  for f in "${ORDER[@]}"; do
    printf "%-48s %s\n" "$f" "${JIDS[$f]}"
  done
} | tee "$out"
echo "Wrote manifest to: $out"
