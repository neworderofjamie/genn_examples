
GENERATED_CODE_DIR	:=vogels_2011_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra 

.PHONY: all clean generated_code

all: simulator

simulator: simulator.cc generated_code
	$(CXX) $(CXXFLAGS) simulator.cc -o simulator -I$(BOB_ROBOTICS_PATH) -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath=$(GENERATED_CODE_DIR)
	
generated_code: 
	$(MAKE) -C $(GENERATED_CODE_DIR)