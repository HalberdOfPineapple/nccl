#!/bin/bash
# Runs every session of one GPU count, in this order: plan 01 calibration (skipped at 8 GPUs, run separately),
# plan 02 scaling, plan 04 link-byte pairs, plan 03 payload sweep, plan 02 GPU-set control (if a config exists).
# Stops at the first failure. Usage: run_chain.sh <gpu_count> <date>
set -e
count=$1
day=$2
# Absolute path, because the next line changes directory.
script_dir=$(cd "$(dirname "$0")" && pwd)
runner=$script_dir/run_allreduce.py
cd "$script_dir/../../scalability"
if [ "$count" != 8 ]; then
  python3 "$runner" selection_calibration/configs/calibration_n${count}.json selection_calibration/results/$day/raw/calibration selection_calibration/results/$day/calibration.jsonl
fi
python3 "$runner" gpu_count_scaling/configs/scaling_n${count}.json gpu_count_scaling/results/$day/raw gpu_count_scaling/results/$day/scaling.jsonl
python3 "$runner" link_bytes/configs/pairs_n${count}.json link_bytes/results/$day/raw link_bytes/results/$day/pairs.jsonl
python3 "$runner" payload_size/configs/sweep_n${count}.json payload_size/results/$day/raw payload_size/results/$day/sweep.jsonl
if [ -f gpu_count_scaling/configs/control_n${count}.json ]; then
  python3 "$runner" gpu_count_scaling/configs/control_n${count}.json gpu_count_scaling/results/$day/raw/control gpu_count_scaling/results/$day/control.jsonl
fi
echo "chain n${count} done"
