#!/usr/bin/env bash
# Build NCCL's contrib example of custom multicast AllReduce kernels (contrib/custom_algos/allreduce)
# for plan 06, without editing the contrib directory.
#
# The kernel headers are compiled unchanged. main.cu gets the plan 06 harness additions from
# patch_custom_mc.py (2 GiB size limit, kernel selection, result check, RESULT lines).
# The patched copy and the binary go to common/bin/ (ignored by git).
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../../../.." && pwd)"
source_dir="${repo_root}/contrib/custom_algos/allreduce"
build_dir="${script_dir}/../bin/custom_mc"

mkdir -p "${build_dir}"
cp "${source_dir}"/*.cuh "${build_dir}/"
python3 "${script_dir}/patch_custom_mc.py" "${source_dir}/main.cu" "${build_dir}/main.cu"

mpi_home="$(mpicc --showme:prefix 2>/dev/null || echo /usr)"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 \
    -gencode arch=compute_90,code=sm_90 \
    -I"${build_dir}" -I"${repo_root}/build/include" -I"${repo_root}/build/include/nccl_device" -I"${repo_root}/src/include" \
    -I"${mpi_home}/include" $(mpicc --showme:compile 2>/dev/null) \
    --expt-relaxed-constexpr --compiler-options "-fPIC" \
    "${build_dir}/main.cu" -o "${build_dir}/allreduce" \
    -L"${repo_root}/build/lib" -lnccl $(mpicc --showme:link 2>/dev/null) -lpthread -lrt -ldl -lcuda
echo "${build_dir}/allreduce"
