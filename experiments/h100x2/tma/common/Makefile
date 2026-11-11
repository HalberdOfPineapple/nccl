NCCL_HOME ?= $(abspath ../../../../build)
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
READ_PRESSURE_INCLUDE = ../../../h100x1/memory_co_runner/common/include
all: bin/tma_congest
# Plan 01: TMA push next to TMA co-runners (bulk stores, bulk reads) and the thread-store writer.
bin/tma_congest: src/tma_congest.cu $(READ_PRESSURE_INCLUDE)/hbm_read_pressure.cuh
	mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=sm_90 -lineinfo --expt-relaxed-constexpr -Xcompiler -pthread -Xptxas=-v -I$(NCCL_HOME)/include -I$(READ_PRESSURE_INCLUDE) $< -o $@ -L$(NCCL_HOME)/lib -lnccl -Xlinker -rpath=$(NCCL_HOME)/lib
.PHONY: all
