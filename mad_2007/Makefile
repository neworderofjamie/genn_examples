GENERATED_CODE_DIR	:=mad_2007_CODE
GENN_PATH :=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: mad_2007

mad_2007: simulator.cc generated_code
	$(CXX) $(CXXFLAGS)  -I$(GENN_PATH) simulator.cc -o mad_2007 -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
