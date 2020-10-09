CUDA_ARCH			?=sm_75
CUDA_PATH			?=/usr/local/cuda

GENERATED_CODE_DIR		:=evidence_accumulation_CODE
BATCH_LEARNING_DIR		:=../../batch_learning
GENN_USERPROJECT_INCLUDE	:=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
NVCC				:= $(CUDA_PATH)/bin/nvcc
NVCCFLAGS			:= -x cu -arch $(CUDA_ARCH) -Xcudafe "--diag_suppress=2937" -std=c++11 -I$(GENN_USERPROJECT_INCLUDE) -I$(BATCH_LEARNING_DIR)
LINKFLAGS			:= -L$(CUDA_PATH)/lib64 -L$(GENERATED_CODE_DIR) -L$(BATCH_LEARNING_DIR) -lcuda -lcudart -lbatch_learning -lrunner -arch $(CUDA_ARCH) --linker-options "-rpath $(GENERATED_CODE_DIR)"

.PHONY: all clean generated_code batch_learning

all: evidence_accumulation

evidence_accumulation: simulator.o generated_code batch_learning
	$(NVCC) -o evidence_accumulation simulator.o $(LINKFLAGS)

batch_learning:
	$(MAKE) -C $(BATCH_LEARNING_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)

-include simulator.d

simulator.d: simulator.cc
	@$(NVCC) -M $(NVCCFLAGS) $< 1> $@

simulator.o: simulator.cc simulator.d
	@$(NVCC) -dc $(NVCCFLAGS) $<

%.d: ;

clean:
	rm -f *.o
	rm -f *.d
	rm -f evidence_accumulation
	$(MAKE) -C $(BATCH_LEARNING_DIR) clean
	$(MAKE) -C $(GENERATED_CODE_DIR) clean
