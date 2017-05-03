EXECUTABLE      := simulator
SOURCES         := simulator.cu

ifdef LIVE
    LINK_FLAGS  += -lpthread -lcaer -lopencv_core -lopencv_highgui -lopencv_imgproc
    CXXFLAGS    += -DLIVE

    ifndef CPU_ONLY
        LINK_FLAGS += -lopencv_gpu
    endif
endif

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk
