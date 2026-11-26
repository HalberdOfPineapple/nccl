#!/usr/bin/env python3
"""Plan 06 harness additions, applied to the build copy of contrib/custom_algos/allreduce/main.cu.

The contrib directory is never edited. Each change looks for an exact upstream text and stops if it is
missing, so an upstream change cannot be patched silently. What is added:

1. Largest message 2 GiB instead of 128 MiB (PERF_MAX_NLINES 2^27 lines of 16 B).
2. CUSTOM_MC_KERNELS (comma list of oneshot, twoshot, poison, nccl; default all) runs only those kernels.
   Needed for link-byte runs, whose counters cover the whole run.
3. CUSTOM_MC_TIMED_ITERS overrides the number of timed calls per size (upstream: 10 at >= 2 MiB).
4. Result check at every measured size: the output buffer is zeroed before each kernel's calls (so a kernel
   that writes nothing cannot pass on an earlier kernel's output), and after the timed calls every element
   must equal nRanks (the input is 1.0 on every GPU). Wrong elements are summed over ranks.
5. One machine-readable line per kernel, size and CTA count, printed by rank 0:
   RESULT kernel=... ranks=... ctas=... threads=... bytes=... iters=... lat_us=... algbw_gbps=... wrong=...
   These key names are read by run_custom_mc.py: do not rename.

Usage: patch_custom_mc.py MAIN_CU_IN MAIN_CU_OUT
"""
import sys
from pathlib import Path


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        sys.exit(f'patch_custom_mc: expected 1 match, found {count}, for:\n{old}')
    return text.replace(old, new)


# Stage 1: file-scope helpers, placed after timedIters().
HELPERS = r'''
// ---- plan 06 additions (patch_custom_mc.py) ----
// Counts elements of buf that differ from expected; used for the result check at every measured size.
__global__ void count_wrong_floats(const float* buf, size_t nFloats, float expected, unsigned long long* wrongCount) {
    unsigned long long localWrong = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nFloats; i += (size_t)gridDim.x * blockDim.x) {
        if (buf[i] != expected) {
            localWrong++;
        }
    }
    if (localWrong > 0) {
        atomicAdd(wrongCount, localWrong);
    }
}

static bool kernelSelected(const char* name) {
    const char* list = getenv("CUSTOM_MC_KERNELS");
    if (list == nullptr || list[0] == '\0') {
        return true;
    }
    std::string padded = std::string(",") + list + ",";
    return padded.find(std::string(",") + name + ",") != std::string::npos;
}

static int timedItersOverride(long nlines) {
    const char* value = getenv("CUSTOM_MC_TIMED_ITERS");
    if (value != nullptr && value[0] != '\0') {
        return atoi(value);
    }
    return timedIters(nlines);
}
// ---- end of plan 06 additions ----
'''

# Stage 2: per-run state at the start of the performance sweep.
SWEEP_SETUP = r'''    CUDACHECK(cudaEventCreate(&evStop));

    // ---- plan 06 additions: kernel selection and result check ----
    const bool runOneshot = kernelSelected("oneshot");
    const bool runTwoshot = kernelSelected("twoshot");
    const bool runPoison = kernelSelected("poison");
    const bool runNccl = kernelSelected("nccl");
    unsigned long long* wrongCountDevice = nullptr;
    CUDACHECK(cudaMalloc((void**)&wrongCountDevice, sizeof(unsigned long long)));

    // Zero an output buffer and make sure every rank has done so before any rank's multicast stores can arrive.
    auto clearOutput = [&](void* buffer, size_t nBytes) {
        CUDACHECK(cudaMemset(buffer, 0, nBytes));
        CUDACHECK(cudaDeviceSynchronize());
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    };

    // Count wrong elements of this rank's output, sum over ranks, print one RESULT line on rank 0.
    auto reportResult = [&](const char* kernelName, const char* ctaText, int threadsPerCta,
                            long measuredBytes, int timedCalls, float latencyUs, const void* output) {
        CUDACHECK(cudaMemset(wrongCountDevice, 0, sizeof(unsigned long long)));
        count_wrong_floats<<<1024, 256, 0, stream>>>((const float*)output, (size_t)measuredBytes / sizeof(float),
                                                    (float)nRanks, wrongCountDevice);
        CUDACHECK(cudaStreamSynchronize(stream));
        unsigned long long wrongCount = 0;
        CUDACHECK(cudaMemcpy(&wrongCount, wrongCountDevice, sizeof(wrongCount), cudaMemcpyDeviceToHost));
        MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &wrongCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
        if (myRank == 0) {
            double algbwGbps = (double)measuredBytes / (latencyUs * 1e-6) / 1e9;
            printf("RESULT kernel=%s ranks=%d ctas=%s threads=%d bytes=%ld iters=%d lat_us=%.3f algbw_gbps=%.3f wrong=%llu\n",
                   kernelName, nRanks, ctaText, threadsPerCta, measuredBytes, timedCalls, latencyUs, algbwGbps, wrongCount);
            fflush(stdout);
        }
    };
    // ---- end of plan 06 additions ----
'''


def apply_patch(text):
    text = replace_once(text, '#define PERF_MAX_NLINES  (1 << 23)', '#define PERF_MAX_NLINES  (1 << 27)')
    text = replace_once(text, '#include <stdio.h>', '#include <stdio.h>\n#include <string>')
    timed_iters_end = '    if (bytes <= (1 << 21)) return  20;\n    return 10;\n}\n'
    text = replace_once(text, timed_iters_end, timed_iters_end + HELPERS)
    text = replace_once(text, '    CUDACHECK(cudaEventCreate(&evStop));\n', SWEEP_SETUP)
    text = replace_once(text, 'int  nTimed    = timedIters(nlines);', 'int  nTimed    = timedItersOverride(nlines);\n            char ctaText[16];\n            snprintf(ctaText, sizeof(ctaText), "%d", nBlocks);')

    # One-shot: select, clear output, report.
    text = replace_once(text,
        '            float oneLat = 0.f;\n            if (hasMC) {\n',
        '            float oneLat = 0.f;\n            if (hasMC && runOneshot) {\n'
        '                clearOutput(recvBuff, (size_t)nlines * sizeof(float4));\n')
    text = replace_once(text,
        '                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &oneLat, 1, MPI_FLOAT,\n                                       MPI_MAX, MPI_COMM_WORLD));\n',
        '                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &oneLat, 1, MPI_FLOAT,\n                                       MPI_MAX, MPI_COMM_WORLD));\n'
        '                reportResult("oneshot", ctaText, MC_AR_MAXTHREADS, nlines * (long)sizeof(float4), nTimed, oneLat, recvBuff);\n')

    # Two-shot with end barrier.
    text = replace_once(text,
        '            float twoLat = 0.f;\n            if (hasMC && twoNlines >= nRanks) {\n',
        '            float twoLat = 0.f;\n            if (hasMC && runTwoshot && twoNlines >= nRanks) {\n'
        '                clearOutput(recvBuff, (size_t)twoNlines * sizeof(float4));\n')
    text = replace_once(text,
        '                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &twoLat, 1, MPI_FLOAT,\n                                       MPI_MAX, MPI_COMM_WORLD));\n',
        '                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &twoLat, 1, MPI_FLOAT,\n                                       MPI_MAX, MPI_COMM_WORLD));\n'
        '                reportResult("twoshot", ctaText, TWOSHOT_MC_MAXTHREADS, twoNlines * (long)sizeof(float4), nTimed, twoLat, recvBuff);\n')

    # Two-shot with poison: its buffers are poisoned (not zero) by upstream code, which the check also catches.
    text = replace_once(text,
        '            float poisLat = 0.f;\n            if (hasMC && twoNlines >= nRanks) {\n',
        '            float poisLat = 0.f;\n            if (hasMC && runPoison && twoNlines >= nRanks) {\n')
    text = replace_once(text,
        '                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &poisLat, 1, MPI_FLOAT,\n                                       MPI_MAX, MPI_COMM_WORLD));\n',
        '                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &poisLat, 1, MPI_FLOAT,\n                                       MPI_MAX, MPI_COMM_WORLD));\n'
        '                // The last timed call wrote poisBufs[(sentIterBase + itersInGraph - 1) % 3].\n'
        '                reportResult("poison", ctaText, TWOSHOT_MC_POISON_MAXTHREADS, twoNlines * (long)sizeof(float4), itersInGraph, poisLat,\n'
        '                             poisBufs[(sentIterBase + itersInGraph - 1) % 3]);\n')

    # NCCL library call on the same windows.
    text = replace_once(text,
        '            float ncclLat = 0.f;\n            {\n',
        '            float ncclLat = 0.f;\n            if (runNccl) {\n'
        '                clearOutput(recvBuff, (size_t)nlines * sizeof(float4));\n')
    text = replace_once(text,
        '                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &ncclLat, 1, MPI_FLOAT,\n                                       MPI_MAX, MPI_COMM_WORLD));\n',
        '                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &ncclLat, 1, MPI_FLOAT,\n                                       MPI_MAX, MPI_COMM_WORLD));\n'
        '                reportResult("nccl", "nccl", 0, nlines * (long)sizeof(float4), nTimed, ncclLat, recvBuff);\n')
    return text


def main():
    source, destination = Path(sys.argv[1]), Path(sys.argv[2])
    destination.write_text(apply_patch(source.read_text()))


if __name__ == '__main__':
    main()
