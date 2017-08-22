EXECUTABLE      := simulator
SOURCES         := simulator.cu

ifndef CPU_ONLY
    SOURCES += $(GENN_PATH)/userproject/include/GeNNHelperKrnls.cu
endif

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk
