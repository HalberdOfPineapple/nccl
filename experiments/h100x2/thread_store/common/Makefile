NCCL_HOME ?= $(abspath ../../../../build)
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
TARGET ?= bin/push_harness
all: $(TARGET) bin/gpc_probe bin/store_congest
$(TARGET): src/push_harness.cu ../../../h100x1/memory_co_runner/common/include/hbm_read_pressure.cuh
	mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=sm_90 -lineinfo --expt-relaxed-constexpr -Xcompiler -pthread -Xptxas=-v -I$(NCCL_HOME)/include -I../../../h100x1/memory_co_runner/common/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib
# Plan 07: SM-to-GPC probe (single GPU, no NCCL) and the store-congestion harness.
bin/gpc_probe: src/gpc_probe.cu
	mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=sm_90 -lineinfo $< -o $@
bin/store_congest: src/store_congest.cu
	mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=sm_90 -lineinfo --expt-relaxed-constexpr -Xcompiler -pthread -Xptxas=-v -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib
.PHONY: all
