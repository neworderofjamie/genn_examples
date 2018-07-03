EXECUTABLE      := simulator
SOURCES         := simulator.cc
INCLUDE_FLAGS   := -I$(BOB_ROBOTICS_PATH)/common
LINK_FLAGS      := `pkg-config --libs opencv`
CXXFLAGS       := -pthread `pkg-config --cflags opencv` 
    
ifdef DVS
    LINK_FLAGS  += -lcaer
    CXXFLAGS    += -DDVS
endif

ifdef CSV
    CXXFLAGS    += -DCSV
endif

ifdef JETSON_POWER
    CXXFLAGS    += -DJETSON_POWER
endif

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk
