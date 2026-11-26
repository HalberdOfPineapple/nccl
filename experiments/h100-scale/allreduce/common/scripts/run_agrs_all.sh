#!/usr/bin/env bash
# AllGather/ReduceScatter series: every remaining step, in order, meant to run detached (tmux) so it survives a lost connection.
# Completed runs are skipped by run_allreduce.py, so the script can be started again after an interruption.
# Run from allgather_reducescatter/scalability/symmetric_multicast/. Usage: run_agrs_all.sh DATE
set -uo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
date_tag="$1"
chain="${script_dir}/run_agrs_chain.sh"
"${chain}" "${date_tag}" 8 selection calibration scaling payload link || exit $?
for gpu_count in 6 4 2; do
    "${chain}" "${date_tag}" "${gpu_count}" selection || exit $?
    if [ "${gpu_count}" = "6" ] || [ "${gpu_count}" = "2" ]; then
        # Short calibration: NCCL's choice and twice it, read from this GPU count's selection runs.
        python3 "${script_dir}/make_configs_agrs.py" "${gpu_count}" --short-calibration "results/${date_tag}/selection.jsonl" || exit $?
    fi
    steps="calibration scaling payload"
    if [ "${gpu_count}" = "4" ]; then
        steps="${steps} link"
    fi
    "${chain}" "${date_tag}" "${gpu_count}" ${steps} || exit $?
done
echo "=== all done $(date -u +%H:%M:%S)"
