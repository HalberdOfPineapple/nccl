#!/bin/bash
# Batch launcher for GB200 NVLS plan 01 (performance validation). Submit on the LOGIN node:
#
#   cd /mnt/home/wli10/nccl
#   P=experiments/gb200x4/nvls/rca/degradation_check/configs/plans
#   sbatch experiments/gb200x4/nvls/common/jobs/plan01.sbatch $P/e1_3.json $P/e1_4.json
#
# Check the request without allocating (beta rejects --test-only, so ask beta_test; same shape):
#   sbatch --test-only -p beta_test --qos=test -t 01:30:00 experiments/gb200x4/nvls/common/jobs/plan01.sbatch $P/smoke.json
#
# What runs where (EMPIRE_AI_BETA_GUIDE.md section 4, CLAUDE_COMPUTE_NODE.md section 4):
#   this file      -> HOST of the allocated node (no CUDA toolkit): settings, paths, one srun step
#   srun step      -> CONTAINER: scripts/run_in_container.sh builds NCCL / nccl-tests if needed,
#                     runs the plans with nccl_runner.py, and exits. No sleeping, no attaching.
#
# Image: a pre-imported squashfs file is used when it exists, so the job does not spend billed GPU
# time pulling ~20 GB (guide section 5). Create it once, on the login node:
#   mkdir -p /ddn/scratch/cornell/wli10/containers
#   enroot import --arch arm64 -o /ddn/scratch/cornell/wli10/containers/pytorch-25.06-py3.sqsh \
#          'docker://nvcr.io#nvidia/pytorch:25.06-py3'
# Without it the job pulls the same pinned tag from nvcr.io.
#
# Resuming: a requeued job keeps its job id and finds its existing session directory (whatever date
# it started), so finished runs are skipped. To continue a session in a NEW job:
#   sbatch --export=ALL,SESSION=<session dir> ...
# Other overrides: CALIBRATION=<json> (calibrated CTA counts), LIMIT_MINUTES=<n> (keep equal to -t
# when squeue is not available inside the job), IMAGE=<image or .sqsh>.
#
#SBATCH -p beta
#SBATCH -A co_jlin4_agenticllm
#SBATCH --qos=standard
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -c 144
#SBATCH -t 01:30:00
#SBATCH -J gb200-nvls-plan01
#SBATCH --requeue
#SBATCH -o /mnt/home/wli10/logs/%x-%j.out
set -euo pipefail

REPOSITORY=/mnt/home/wli10/nccl
COMMON="$REPOSITORY/experiments/gb200x4/nvls/common"
RESULTS="$REPOSITORY/experiments/gb200x4/nvls/rca/degradation_check/results"
SQUASHFS_IMAGE=/ddn/scratch/cornell/wli10/containers/pytorch-25.06-py3.sqsh
REGISTRY_IMAGE='nvcr.io#nvidia/pytorch:25.06-py3'
MOUNTS=/mnt/home/wli10:/mnt/home/wli10,/projects/co/jlin4/agenticllm:/projects/co/jlin4/agenticllm,/ddn/scratch/cornell/wli10:/ddn/scratch/cornell/wli10
CALIBRATION="${CALIBRATION:-$REPOSITORY/experiments/gb200x4/nvls/rca/degradation_check/configs/calibrated_ctas.json}"

[ $# -gt 0 ] || { echo "usage: sbatch plan01.sbatch PLAN.json..." >&2; exit 2; }

if [ -z "${IMAGE:-}" ]; then
  if [ -f "$SQUASHFS_IMAGE" ]; then
    IMAGE="$SQUASHFS_IMAGE"
  else
    IMAGE="$REGISTRY_IMAGE"
  fi
fi

# Plans may be given relative to the repository root.
plans=()
for plan in "$@"; do
  case "$plan" in
    /*) plans+=("$plan") ;;
    *) plans+=("$REPOSITORY/$plan") ;;
  esac
done

# Session directory: reuse this job id's directory if a previous attempt (requeue) created one.
if [ -z "${SESSION:-}" ]; then
  SESSION=$(ls -d "$RESULTS"/*/job_"$SLURM_JOB_ID" 2>/dev/null | head -1 || true)
  [ -n "$SESSION" ] || SESSION="$RESULTS/$(date -u +%Y-%m-%d)/job_${SLURM_JOB_ID}"
fi
mkdir -p "$SESSION"

# Time budget for starting new runs: the job's wall limit minus 15 minutes for image start,
# builds and the last run. Read from Slurm when squeue exists here (so `sbatch -t` changes it);
# otherwise from LIMIT_MINUTES (default 90 = the #SBATCH -t above).
limitMinutes=""
if command -v squeue > /dev/null 2>&1; then
  limitMinutes=$(squeue -h -j "$SLURM_JOB_ID" -o %l 2>/dev/null | awk -F'[-:]' '{ if (NF==4) print $1*1440+$2*60+$3; else if (NF==3) print $1*60+$2; else if (NF==2) print $1; }' || true)
fi
budgetMinutes=$(( ${limitMinutes:-${LIMIT_MINUTES:-90}} - 15 ))

echo "job $SLURM_JOB_ID (restart count ${SLURM_RESTART_COUNT:-0}) on $(hostname) at $(date -u +%FT%TZ)"
echo "image $IMAGE; session $SESSION; budget ${budgetMinutes} min; plans: ${plans[*]}"

calibrationArguments=()
[ -f "$CALIBRATION" ] && calibrationArguments=(--calibration "$CALIBRATION")

# One containerised step. CPUs are passed explicitly: depending on the Slurm version a step does not
# inherit the job's -c, and the NCCL build (~50 CPU-minutes) would then run on one core.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-144}" \
     --container-image="$IMAGE" \
     --container-mounts="$MOUNTS" --container-mount-home \
     --container-workdir="$REPOSITORY" \
     --container-env=PROJECT_DIR,HF_HOME \
     bash "$COMMON/scripts/run_in_container.sh" --session "$SESSION" --minutes "$budgetMinutes" \
          "${calibrationArguments[@]}" "${plans[@]}"

echo "job $SLURM_JOB_ID done at $(date -u +%FT%TZ): $SESSION"
