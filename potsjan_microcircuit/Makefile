GENERATED_CODE_DIR	:=potjans_microcircuit_CODE
GENN_PATH :=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: potjans_microcircuit

potjans_microcircuit: simulator.cc generated_code
	$(CXX) $(CXXFLAGS)  -I$(GENN_PATH) simulator.cc -o potjans_microcircuit -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

potjans_microcircuit_shared_library: simulator_shared_library.cc generated_code
	$(CXX) $(CXXFLAGS) -I$(GENN_PATH) simulator_shared_library.cc -ldl -o potjans_microcircuit_shared_library

potjans_microcircuit_live_shared_library: simulator_live_shared_library.cc generated_code
	$(CXX) $(CXXFLAGS) -I$(GENN_PATH) simulator_live_shared_library.cc -pthread `pkg-config --libs --cflags opencv` -ldl -o potjans_microcircuit_live_shared_library
generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
