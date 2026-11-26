#!/usr/bin/env bash
# AllGather/ReduceScatter series: the fixed-CTA cells added after calibration (Ring and classic NVLS at 64 CTAs), scaling then payload,
# 8 GPUs first. Completed runs are skipped, so the script can be restarted. Run from symmetric_multicast/. Usage: run_agrs_fixed.sh DATE
set -uo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for gpu_count in 8 6 4 2; do
    "${script_dir}/run_agrs_chain.sh" "$1" "${gpu_count}" scaling payload || exit $?
done
echo "=== fixed-CTA cells done $(date -u +%H:%M:%S)"
