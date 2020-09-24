CUDA_ARCH			?=sm_75
CUDA_PATH			?=/usr/local/cuda
OBJECTS 			:=batch_learning.o

DEPS				:= $(OBJECTS:.o=.d)
NVCC				:= $(CUDA_PATH)/bin/nvcc
NVCCFLAGS			:= -x cu -arch $(CUDA_ARCH) -Xcudafe "--diag_suppress=2937" -std=c++11

.PHONY: all clean generated_code

all: libbatch_learning.a

libbatch_learning.a: $(OBJECTS)
	@$(AR) $(ARFLAGS) $@ $(OBJECTS)

-include $(DEPS)

%.d: %.cc
	@$(NVCC) -M $(NVCCFLAGS) $< 1> $@

%.o: %.cc %.d
	@$(NVCC) -dc $(NVCCFLAGS) $<

%.d: ;

clean:
	rm -f *.o
	rm -f *.d
	rm -f libbatch_learning.a
