GENERATED_CODE_DIR		:=s_mnist_CODE
GENN_USERPROJECT_INCLUDE	:=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
CXXFLAGS 			+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: s_mnist

s_mnist: simulator.cc generated_code
	$(CXX) $(CXXFLAGS)  -I$(GENN_USERPROJECT_INCLUDE) simulator.cc -o s_mnist -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
