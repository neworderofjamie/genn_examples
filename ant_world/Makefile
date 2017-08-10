EXECUTABLE      := ant_world
SOURCES         := ant_world.cc render_mesh.cc route.cc snapshot_processor.cc world.cc $(GENN_PATH)/userproject/include/GeNNHelperKrnls.cu
LINK_FLAGS      := -lglfw -lGL -lGLU -lGLEW  -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
CXXFLAGS        := -pthread -Wall -Wpedantic -Wextra

ifdef RECORD_SPIKES
    CXXFLAGS += -DRECORD_SPIKES
endif

ifdef RECORD_TERMINAL_SYNAPSE_STATE
    CXXFLAGS += -DRECORD_TERMINAL_SYNAPSE_STATE
endif

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk