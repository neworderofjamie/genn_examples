GENERATED_CODE_DIR	:=benchmark_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: benchmark

benchmark: simulator.cc generated_code
	$(CXX) $(CXXFLAGS)  simulator.cc -o benchmark -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
