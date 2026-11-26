#!/bin/bash
# Container side of every GB200 NVLS experiment session (batch or interactive).
#
# Runs INSIDE nvcr.io/nvidia/pytorch:25.06-py3. It is started either
#   - by the batch launcher (../jobs/plan01.sbatch) through `srun --container-image=...`, or
#   - in an interactive beta_test session through .vscode/in_container.sh.
# It never calls srun (the image has no Slurm client).
#
# Steps: check we are in the container -> clean PATH -> (re)build NCCL and nccl-tests if needed
#        -> run each plan in order with nccl_runner.py into one session directory.
#
# Usage: run_in_container.sh --session <dir> [--minutes <budget>] [--calibration <json>] <plan.json>...
set -euo pipefail

[ -x /usr/local/cuda/bin/nvcc ] || { echo "run_in_container.sh must run inside the container" >&2; exit 1; }

# The submitting shell's PATH (conda, ~/envs/buildtools) is forwarded by sbatch and by enroot;
# drop the host-only entries so python, make and nvcc are the container's.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -vE '^/mnt/home/wli10/(envs|miniconda3|\.local|\.cargo)' | paste -sd:)
export PATH="/usr/local/cuda/bin:$PATH"
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda

REPOSITORY=/mnt/home/wli10/nccl
NCCL_BUILD="$REPOSITORY/build-gb200"
NCCL_TESTS=/mnt/home/wli10/nccl-tests
GENCODE="-gencode=arch=compute_100,code=sm_100"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/usr/bin/python3

session=""
minutes=""
calibration=""
plans=()
while [ $# -gt 0 ]; do
  case "$1" in
    --session) session="$2"; shift 2 ;;
    --minutes) minutes="$2"; shift 2 ;;
    --calibration) calibration="$2"; shift 2 ;;
    *) plans+=("$1"); shift ;;
  esac
done
[ -n "$session" ] && [ ${#plans[@]} -gt 0 ] || { echo "usage: $0 --session DIR [--minutes M] [--calibration JSON] PLAN..." >&2; exit 2; }
mkdir -p "$session/raw"
startSeconds=$(date +%s)
# nproc counts the CPUs this step may use (cgroup and affinity); SLURM_CPUS_ON_NODE would report the
# whole node even when the step got fewer.
jobs="$(nproc)"

# Build stage: make is incremental, so an unchanged tree costs seconds. A build failure stops the
# session before any GPU run, because results from a stale library would be mislabelled.
buildSoftware() {
  echo "== build NCCL ($(date -u +%H:%M:%S))"
  make -C "$REPOSITORY" -j "$jobs" src.build CUDA_HOME=/usr/local/cuda BUILDDIR="$NCCL_BUILD" \
       NVCC_GENCODE="$GENCODE" > "$session/raw/build_nccl.log" 2>&1 \
    || { tail -30 "$session/raw/build_nccl.log"; exit 1; }
  if [ ! -d "$NCCL_TESTS" ]; then
    git clone -q https://github.com/NVIDIA/nccl-tests.git "$NCCL_TESTS"
  fi
  echo "== build nccl-tests ($(date -u +%H:%M:%S))"
  make -C "$NCCL_TESTS" -j "$jobs" BUILDDIR=build-gb200 CUDA_HOME=/usr/local/cuda NCCL_HOME="$NCCL_BUILD" \
       NVCC_GENCODE="$GENCODE" > "$session/raw/build_nccl_tests.log" 2>&1 \
    || { tail -30 "$session/raw/build_nccl_tests.log"; exit 1; }
}

# Plan stage: plans run in the given order; each gets the minutes left in the budget.
runPlans() {
  for plan in "${plans[@]}"; do
    arguments=("$plan" --session "$session")
    [ -n "$calibration" ] && arguments+=(--calibration "$calibration")
    if [ -n "$minutes" ]; then
      elapsedMinutes=$(( ($(date +%s) - startSeconds) / 60 ))
      leftMinutes=$(( minutes - elapsedMinutes ))
      [ "$leftMinutes" -gt 0 ] || { echo "== budget used up before $plan"; break; }
      arguments+=(--stop-after-minutes "$leftMinutes")
    fi
    echo "== plan $plan ($(date -u +%H:%M:%S))"
    "$PYTHON" "$SCRIPTS/nccl_runner.py" "${arguments[@]}"
  done
}

buildSoftware
runPlans
echo "== session done ($(date -u +%H:%M:%S)): $session"
