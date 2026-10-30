// HBM read pressure with a target-rate knob, for use as one CTA role inside a fused kernel.
//
// One thread per CTA asks the Hopper bulk-copy unit (TMA) to copy fixed-size chunks from a
// large global buffer into this CTA's shared memory. Several chunks are in flight at once.
// The data is not used; it only creates memory traffic. One thread is enough because the
// copy unit, not the thread, moves the bytes, so the role needs few SMs and few registers.
//
// Knobs:
//   chunkBytes, chunksInFlight  how many bytes this CTA keeps in flight (calibrated once)
//   chunkIntervalPs             time between chunk starts for this CTA; 0 = as fast as possible
// The interval is the rate knob: CTA count and in-flight bytes stay fixed while it changes.
//
// Requirements: sm_90; at least two warps per CTA; the caller gives chunksInFlight × chunkBytes
// bytes of shared memory (128-byte aligned), chunksInFlight 8-byte barriers, a shared int set
// to 0 before both roles start, and a stop flag in global memory.
//
// Use: after a CTA barrier, thread 0 calls readUntilStopped and one thread of another warp
// calls forwardStopFlag. Reading the global flag in the copy loop cost a memory round trip
// (about 0.5 µs) per chunk and limited each SM to about one chunk at a time, so a second
// warp watches the global flag and copies it into shared memory.
#pragma once
#include <cstdint>
#include <cuda/atomic>

struct HbmReadSettings {
  const uint8_t* workingSet;              // global buffer, much larger than L2
  unsigned long long workingSetBytes;
  int pressureCtas;                       // the buffer is split into one slice per pressure CTA
  unsigned chunkBytes;                    // multiple of 16, at most 1 MiB - 1
  int chunksInFlight;
  unsigned long long chunkIntervalPs;     // per CTA; 0 means no rate limit
};

struct HbmReadResult {
  unsigned long long startNs;
  unsigned long long endNs;
  unsigned long long chunksRead;          // includes chunks drained after the stop signal
  unsigned long long checksum;            // sum of the first word of every chunk read
};

namespace hbm_read_pressure {

__device__ __forceinline__ unsigned long long timerNs() {
  unsigned long long timestampNs;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestampNs));
  return timestampNs;
}

__device__ __forceinline__ uint32_t sharedAddress(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void initializeBarrier(uint64_t* barrier) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sharedAddress(barrier)) : "memory");
}

__device__ __forceinline__ void invalidateBarrier(uint64_t* barrier) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(sharedAddress(barrier)) : "memory");
}

// Start one bulk copy; the barrier completes when all its bytes have arrived.
__device__ __forceinline__ void startChunkCopy(void* sharedDestination, const void* globalSource, unsigned bytes, uint64_t* barrier) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sharedAddress(barrier)), "r"(bytes) : "memory");
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sharedAddress(sharedDestination)), "l"(globalSource), "r"(bytes), "r"(sharedAddress(barrier)) : "memory");
}

__device__ __forceinline__ void waitForChunk(uint64_t* barrier, unsigned phase) {
  asm volatile("{\n"
               "  .reg .pred done;\n"
               "WAIT_%=:\n"
               "  mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
               "  @!done bra WAIT_%=;\n"
               "}\n" :: "r"(sharedAddress(barrier)), "r"(phase) : "memory");
}

// Our normal read of a slot must finish before the copy unit writes that slot again.
__device__ __forceinline__ void orderSharedReadBeforeNextCopy() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ bool stopRequested(volatile int* sharedStopFlag) {
  return *sharedStopFlag != 0;
}

// Run by one thread of a warp other than the copy thread's warp.
__device__ void forwardStopFlag(int* globalStopFlag, volatile int* sharedStopFlag) {
  while (cuda::atomic_ref<int, cuda::thread_scope_system>(*globalStopFlag).load(cuda::memory_order_acquire) == 0) {
    __nanosleep(256);
  }
  *sharedStopFlag = 1;
}

// Rate limit: chunk k of this CTA may start at startNs + k × interval, never earlier.
// A late chunk starts at once, so a CTA that fell behind catches up only as far as
// chunksInFlight allows; achieved rate below target therefore shows lasting slowdown.
__device__ __forceinline__ void waitForScheduledStart(const HbmReadSettings& settings, unsigned long long startNs, unsigned long long chunkIndex) {
  if (settings.chunkIntervalPs == 0) {
    return;
  }
  unsigned long long scheduledNs = startNs + chunkIndex * settings.chunkIntervalPs / 1000;
  while (timerNs() < scheduledNs) {
  }
}

// Run by exactly one thread of a pressure CTA until stopFlag becomes non-zero.
// Every started copy is waited for before returning, so shared memory is free afterwards.
__device__ void readUntilStopped(const HbmReadSettings& settings, int pressureIndex, uint8_t* sharedSlots, uint64_t* sharedBarriers, volatile int* sharedStopFlag, HbmReadResult& result) {
  const unsigned long long sliceChunks = settings.workingSetBytes / settings.chunkBytes / settings.pressureCtas;
  const uint8_t* slice = settings.workingSet + pressureIndex * sliceChunks * settings.chunkBytes;
  for (int slot = 0; slot < settings.chunksInFlight; ++slot) {
    initializeBarrier(&sharedBarriers[slot]);
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  const unsigned long long startNs = timerNs();
  unsigned long long chunksStarted = 0;
  unsigned long long chunksRead = 0;
  unsigned long long checksum = 0;
  // Fill every slot, then start a new chunk each time one arrives.
  for (int slot = 0; slot < settings.chunksInFlight; ++slot) {
    waitForScheduledStart(settings, startNs, chunksStarted);
    startChunkCopy(sharedSlots + size_t(slot) * settings.chunkBytes, slice + (chunksStarted % sliceChunks) * settings.chunkBytes, settings.chunkBytes, &sharedBarriers[slot]);
    ++chunksStarted;
  }
  bool stopping = false;
  while (chunksRead < chunksStarted) {
    const int slot = chunksRead % settings.chunksInFlight;
    const unsigned phase = (chunksRead / settings.chunksInFlight) & 1;
    waitForChunk(&sharedBarriers[slot], phase);
    // Reading one word proves the bytes arrived; the buffer holds a known pattern.
    checksum += *reinterpret_cast<volatile uint32_t*>(sharedSlots + size_t(slot) * settings.chunkBytes);
    ++chunksRead;
    if (!stopping) {
      stopping = stopRequested(sharedStopFlag);
    }
    if (stopping) {
      continue;
    }
    orderSharedReadBeforeNextCopy();
    waitForScheduledStart(settings, startNs, chunksStarted);
    startChunkCopy(sharedSlots + size_t(slot) * settings.chunkBytes, slice + (chunksStarted % sliceChunks) * settings.chunkBytes, settings.chunkBytes, &sharedBarriers[slot]);
    ++chunksStarted;
  }
  for (int slot = 0; slot < settings.chunksInFlight; ++slot) {
    invalidateBarrier(&sharedBarriers[slot]);
  }
  result.startNs = startNs;
  result.endNs = timerNs();
  result.chunksRead = chunksRead;
  result.checksum = checksum;
}

}  // namespace hbm_read_pressure
