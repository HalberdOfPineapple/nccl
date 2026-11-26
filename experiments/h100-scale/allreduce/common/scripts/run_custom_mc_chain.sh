#!/usr/bin/env bash
# Plan 06: E6.0 check, E6.1 CTA sweep, flat CTA counts, E6.2 payload runs and (at 4 GPUs) E6.3 link pairs, for each GPU count given.
# Run from scalability/custom_mc_kernels/. Usage: run_custom_mc_chain.sh DATE GPU_COUNT [GPU_COUNT ...]
set -uo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
date_dir="results/$1"
shift
for gpu_count in "$@"; do
    echo "=== ${gpu_count} GPUs, $(date -u +%H:%M:%S)"
    python3 "${script_dir}/run_custom_mc.py" "configs/check_n${gpu_count}.json" "${date_dir}/raw" "${date_dir}/check.jsonl" || exit $?
    if ! tail -n 1 "${date_dir}/check.jsonl" | grep -q '"valid": true'; then
        echo "check at ${gpu_count} GPUs not valid; skipping this GPU count"
        continue
    fi
    python3 "${script_dir}/run_custom_mc.py" "configs/cta_n${gpu_count}.json" "${date_dir}/raw" "${date_dir}/cta.jsonl" || exit $?
    python3 "${script_dir}/summarize_custom_mc.py" "${date_dir}/summary.csv" "${date_dir}"/check.jsonl "${date_dir}"/cta.jsonl > /dev/null
    read -r oneshot twoshot poison < <(python3 "${script_dir}/summarize_custom_mc.py" --flat-values "${date_dir}/summary.csv" "${gpu_count}")
    echo "flat CTA counts at ${gpu_count} GPUs: oneshot ${oneshot}, twoshot ${twoshot}, poison ${poison}"
    python3 "${script_dir}/make_configs_custom_mc.py" --flat "${gpu_count}" "${oneshot}" "${twoshot}" "${poison}"
    python3 "${script_dir}/run_custom_mc.py" "configs/flat_n${gpu_count}.json" "${date_dir}/raw" "${date_dir}/flat.jsonl" || exit $?
    if [ "${gpu_count}" = "4" ]; then
        python3 "${script_dir}/run_custom_mc.py" "configs/link_n${gpu_count}.json" "${date_dir}/raw" "${date_dir}/link.jsonl" || exit $?
    fi
done
echo "=== chain done $(date -u +%H:%M:%S)"
