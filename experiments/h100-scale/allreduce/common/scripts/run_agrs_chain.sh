#!/usr/bin/env bash
# AllGather/ReduceScatter series: runs the given steps for one GPU count, in order, with run_allreduce.py.
# Run from allgather_reducescatter/scalability/symmetric_multicast/. Usage: run_agrs_chain.sh DATE GPU_COUNT STEP [STEP ...]
# Steps: selection calibration scaling payload link (each reads configs/<step>_n<N>.json, appends results/DATE/<step>.jsonl).
set -uo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
date_dir="results/$1"
gpu_count="$2"
shift 2
for step in "$@"; do
    echo "=== ${step} at ${gpu_count} GPUs, $(date -u +%H:%M:%S)"
    python3 "${script_dir}/run_allreduce.py" "configs/${step}_n${gpu_count}.json" "${date_dir}/raw" "${date_dir}/${step}.jsonl" || exit $?
done
echo "=== done $(date -u +%H:%M:%S)"
