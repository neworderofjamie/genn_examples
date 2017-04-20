EXECUTABLE      := simulator
SOURCES         := simulator.cc
LINK_FLAGS      := -lopencv_core -lopencv_highgui -lopencv_imgproc
ifndef CPU_ONLY
    LINK_FLAGS += -lopencv_gpu
endif
include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk
