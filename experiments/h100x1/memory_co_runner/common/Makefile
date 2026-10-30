CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
TARGET ?= bin/hbm_read_calibration
all: $(TARGET)
$(TARGET): src/hbm_read_calibration.cu include/hbm_read_pressure.cuh
	mkdir -p bin
	$(NVCC) -O3 -std=c++17 -arch=sm_90 -lineinfo -Xptxas=-v $< -o $@
.PHONY: all
