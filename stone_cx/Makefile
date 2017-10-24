EXECUTABLE      := simulator
SOURCES         := simulator.cc
LINK_FLAGS	:= -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui

CPU_ONLY=1

ifdef RECORD_ELECTROPHYS
    CXXFLAGS += -DRECORD_ELECTROPHYS
endif

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk
