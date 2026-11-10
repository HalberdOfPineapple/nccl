NCCL_HOME ?= $(abspath ../../../../build)
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
all: bin/mm_mix bin/multicast_probe
# Plan 01: st_multicast, ld_reduce and plain ld/st roles on four GPUs.
bin/mm_mix: src/mm_mix.cu
	mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=sm_90 -lineinfo --expt-relaxed-constexpr --extended-lambda -Xcompiler -pthread -Xptxas=-v -I$(NCCL_HOME)/include $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib
# Plan 01 E1.0: reads the multicast attribute per GPU.
bin/multicast_probe: src/multicast_probe.cu
	mkdir -p bin
	$(NVCC) -std=c++17 $< -o $@ -lcuda
.PHONY: all
