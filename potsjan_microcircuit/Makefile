
GENERATED_CODE_DIR	:=potjans_microcircuit_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra -I$(CUDA_PATH)/include

.PHONY: all clean generated_code

all: simulator

simulator: simulator.cc generated_code
	$(CXX) $(CXXFLAGS) simulator.cc -o simulator -I$(BOB_ROBOTICS_PATH) -L$(GENERATED_CODE_DIR) -L$(CUDA_PATH)/lib64 -lrunner -lcudart -Wl,-rpath=$(GENERATED_CODE_DIR)

generated_code: 
	$(MAKE) -C $(GENERATED_CODE_DIR)