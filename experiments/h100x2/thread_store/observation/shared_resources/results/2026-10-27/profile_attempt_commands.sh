#!/usr/bin/env bash
# Run after the primary campaign; do not mix profiler runs with the paired timing stream.
set -euo pipefail
common_directory=$(cd "$(dirname "$0")/.." && pwd)
experiment_directory="$common_directory/../observation/shared_resources"
result_directory="$experiment_directory/results/2026-10-27"
profile_directory="$result_directory/profiles"
mkdir -p "$profile_directory"
for variant in healthy compute source_memory destination_memory vector_source_memory vector_destination_memory; do
  config="$experiment_directory/configs/profile_${variant}.cfg"
  python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); import run_one;run_one.ensure_idle()' "$common_directory/scripts"
  timeout --kill-after=5s 60s nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=false --output="$profile_directory/nsys_${variant}" "$common_directory/bin/push_harness" "$config" "$profile_directory/nsys_${variant}.json" > "$profile_directory/nsys_${variant}.log" 2>&1 || printf 'nsys failed for %s\n' "$variant"
  if [[ -f "$profile_directory/nsys_${variant}.nsys-rep" ]]; then
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv --output "$profile_directory/nsys_${variant}_stats" "$profile_directory/nsys_${variant}.nsys-rep" > "$profile_directory/nsys_${variant}_stats.log" 2>&1 || true
  fi
  for device in 4 5; do
    python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); import run_one;run_one.ensure_idle()' "$common_directory/scripts"
    # Profile only one endpoint while its peer remains executable on the other GPU.
    # Application replay reinitializes state for each pass. Overlap and correctness
    # are validated from each pass's output/log, rather than assumed from the mode.
    timeout --kill-after=5s 90s ncu --devices "$device" --replay-mode application --cache-control none --clock-control none --pipeline-boost-state dynamic --import-source yes --source-folders "$common_directory/src" --kernel-name regex:fusedPush --launch-skip 1 --launch-count 1 --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors.sum,smsp__warps_eligible.avg.per_cycle_active,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__inst_executed_pipe_fma.sum --export "$profile_directory/ncu_${variant}_gpu${device}" "$common_directory/bin/push_harness" "$config" "$profile_directory/ncu_${variant}_gpu${device}.json" > "$profile_directory/ncu_${variant}_gpu${device}.log" 2>&1 || printf 'ncu failed for %s GPU %s\n' "$variant" "$device"
    if [[ -f "$profile_directory/ncu_${variant}_gpu${device}.ncu-rep" ]]; then
      ncu --import "$profile_directory/ncu_${variant}_gpu${device}.ncu-rep" --csv --page raw > "$profile_directory/ncu_${variant}_gpu${device}.csv" 2> "$profile_directory/ncu_${variant}_gpu${device}_export.log" || true
    fi
  done
done

# PC-correlated samples are separate from aggregate hardware counters.
for variant in healthy compute source_memory destination_memory vector_source_memory vector_destination_memory; do
  config="$experiment_directory/configs/profile_${variant}.cfg"
  python3 -c 'import sys; sys.path.insert(0,sys.argv[1]); import run_one;run_one.ensure_idle()' "$common_directory/scripts"
  timeout --kill-after=5s 90s ncu --devices 4 --replay-mode application --cache-control none --clock-control none --pipeline-boost-state dynamic --import-source yes --source-folders "$common_directory/src" --kernel-name regex:fusedPush --launch-skip 1 --launch-count 1 --warp-sampling-max-passes 1 --metrics smsp__pcsamp_warps_issue_stalled_long_scoreboard,smsp__pcsamp_warps_issue_stalled_lg_throttle,smsp__pcsamp_warps_issue_stalled_membar,smsp__pcsamp_warps_issue_stalled_barrier --export "$profile_directory/pc_${variant}" "$common_directory/bin/push_harness" "$config" "$profile_directory/pc_${variant}.json" > "$profile_directory/pc_${variant}.log" 2>&1 || printf 'PC sampling failed for %s\n' "$variant"
  if [[ -f "$profile_directory/pc_${variant}.ncu-rep" ]]; then
    ncu --import "$profile_directory/pc_${variant}.ncu-rep" --csv --page source --print-source cuda,sass > "$profile_directory/pc_${variant}.csv" 2> "$profile_directory/pc_${variant}_export.log" || true
  fi
done
