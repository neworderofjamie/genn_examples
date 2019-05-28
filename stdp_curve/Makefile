GENERATED_CODE_DIR	:=stdp_curve_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: stdp_curve

stdp_curve: simulator.cc generated_code
	$(CXX) $(CXXFLAGS)  simulator.cc -o stdp_curve -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
