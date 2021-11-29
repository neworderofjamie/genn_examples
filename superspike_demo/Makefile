OPENCV_PACKAGE                  :=opencv4
GENERATED_CODE_DIR		:=superspike_demo_CODE
GENN_USERPROJECT_INCLUDE	:=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
CXXFLAGS 			+=-std=c++11 -Wall -Wpedantic -Wextra

ifneq ("$(wildcard /sys/module/tegra_fuse/parameters/tegra_chip_id)","")
    TEGRA_CHIP_ID		:=$(shell cat /sys/module/tegra_fuse/parameters/tegra_chip_id)
else
    TEGRA_CHIP_ID		:=-1
endif

.PHONY: all clean generated_code

all: superspike_demo

superspike_demo: simulator.cc generated_code
	$(CXX) $(CXXFLAGS)  -I$(GENN_USERPROJECT_INCLUDE) simulator.cc -o superspike_demo -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

superspike_demo_live: simulator_live.cc generated_code
	$(CXX) $(CXXFLAGS) `pkg-config --cflags $(OPENCV_PACKAGE)` simulator_live.cc -DTEGRA_CHIP_ID=$(TEGRA_CHIP_ID) -o superspike_demo_live -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR) -pthread `pkg-config --libs $(OPENCV_PACKAGE)` 

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
