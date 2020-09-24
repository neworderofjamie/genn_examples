CUDA_ARCH			?=sm_75
CUDA_PATH			?=/usr/local/cuda

GENERATED_CODE_DIR		:=superspike_demo_CODE
BATCH_LEARNING_DIR		:=../batch_learning
GENN_USERPROJECT_INCLUDE	:=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
NVCC				:= $(CUDA_PATH)/bin/nvcc
NVCCFLAGS			:= -x cu -arch $(CUDA_ARCH) -Xcudafe "--diag_suppress=2937" -std=c++11 -I$(GENN_USERPROJECT_INCLUDE) -I$(BATCH_LEARNING_DIR)
LINKFLAGS			:= -L$(CUDA_PATH)/lib64 -L$(GENERATED_CODE_DIR) -L$(BATCH_LEARNING_DIR) -lcuda -lcudart -lbatch_learning -lrunner -arch $(CUDA_ARCH) --linker-options "-rpath $(GENERATED_CODE_DIR)"

.PHONY: all clean generated_code batch_learning

all: superspike_demo

superspike_demo: simulator.o generated_code batch_learning
	$(NVCC) -o superspike_demo simulator.o $(LINKFLAGS)

simulator.o: simulator.cc
	@$(NVCC) $(NVCCFLAGS) -dc simulator.cc

batch_learning:
	$(MAKE) -C $(BATCH_LEARNING_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
