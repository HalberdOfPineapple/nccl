#!/usr/bin/env bash
# Each pair is two separately initialized harness invocations, sharing environmental snapshots.
set -euo pipefail
output_directory=${1:?output directory}
pair_count=${2:?pair count}
healthy_config=${3:?healthy config}
perturbed_config=${4:?perturbed config}
label=${5:?label}
script_directory=$(cd "$(dirname "$0")" && pwd)
for ((pair_index=1; pair_index<=pair_count; pair_index++)); do
  if [[ ! -f "$output_directory/${label}_H_${pair_index}.json" ]]; then
    python3 "$script_directory/run_one.py" "$healthy_config" "$output_directory/${label}_H_${pair_index}.json"
  fi
  if [[ ! -f "$output_directory/${label}_P_${pair_index}.json" ]]; then
    python3 "$script_directory/run_one.py" "$perturbed_config" "$output_directory/${label}_P_${pair_index}.json"
  fi
done
