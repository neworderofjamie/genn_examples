GENERATED_CODE_DIR	:=potjans_microcircuit_CODE
CXXFLAGS 		+=-std=c++14 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: potjans_microcircuit

potjans_microcircuit: simulator.cc generated_code
	$(CXX) $(CXXFLAGS) -I$(BOB_ROBOTICS_PATH) -I$(BOB_ROBOTICS_PATH)/third_party/plog/include simulator.cc -o potjans_microcircuit -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
