EXECUTABLE      := simulator
SOURCES         := simulator.cc
INCLUDE_FLAGS   := -I$(GENN_ROBOTICS_PATH)/common

ifdef MPI_ENABLE
    SIM_CODE := va_benchmark_${OMPI_COMM_WORLD_RANK}_CODE
    OBJECT_PATH := $(SIM_CODE)/
    EXECUTABLE := simulator_${OMPI_COMM_WORLD_RANK}
    INCLUDE_FLAGS += -DDEFINITIONS_HEADER='"$(SIM_CODE)/definitions.h"'
endif

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk
