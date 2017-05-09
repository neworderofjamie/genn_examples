EXECUTABLE      := simulator
SOURCES         := simulator.cu

LINK_FLAGS      += -lpthread -lopencv_core -lopencv_highgui -lopencv_imgproc

ifndef CPU_ONLY
    LINK_FLAGS += -lopencv_gpu
endif
    
ifdef LIVE
    LINK_FLAGS  += -lcaer
    CXXFLAGS    += -DLIVE
endif

ifdef JETSON_POWER
    CXXFLAGS    += -DJETSON_POWER
endif

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk
