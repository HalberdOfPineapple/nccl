#!/usr/bin/env bash
# Validated timeline capture. Hardware counter attempts are retained separately;
# they did not produce valid ranges on this installed tool/workload combination.
set -euo pipefail
common_directory=$(cd "$(dirname "$0")/.." && pwd)
experiment_directory="$common_directory/../observation/shared_resources"
profile_directory="$experiment_directory/results/2026-10-27/profiles"
mkdir -p "$profile_directory"
for variant in healthy compute source_memory destination_memory vector_source_memory vector_destination_memory; do
  config="$experiment_directory/configs/profile_${variant}.cfg"
  output_prefix="$profile_directory/nsys_range_${variant}"
  if [[ -f "$output_prefix.nsys-rep" && -f "$output_prefix.json" ]]; then
    continue
  fi
  python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); import run_one;run_one.ensure_idle()' "$common_directory/scripts"
  timeout --kill-after=5s 60s nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=false --output="$output_prefix" "$common_directory/bin/push_harness" "$config" "$output_prefix.json" --profile-range > "$output_prefix.log" 2>&1
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv --output "${output_prefix}_stats" "$output_prefix.nsys-rep" > "${output_prefix}_stats.log" 2>&1
  printf 'Timeline captured: %s\n' "$variant"
done
