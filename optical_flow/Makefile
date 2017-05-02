EXECUTABLE      := simulator
SOURCES         := simulator.cu

ifdef LIVE
    LINK_FLAGS  += -lcaer
    CXXFLAGS    += -DLIVE
endif

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk
