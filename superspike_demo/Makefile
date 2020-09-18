GENERATED_CODE_DIR		:=superspike_demo_CODE
BATCH_LEARNING_DIR		:=../batch_learning
GENN_USERPROJECT_INCLUDE	:=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
CXXFLAGS 			+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: superspike_demo batch_learning

superspike_demo: simulator.cc generated_code batch_learning
	$(CXX) $(CXXFLAGS)  -I$(GENN_USERPROJECT_INCLUDE) -I$(BATCH_LEARNING_DIR) simulator.cc -o superspike_demo -L$(CUDA_PATH)/lib64 -L$(GENERATED_CODE_DIR) -L$(BATCH_LEARNING_DIR) -lbatch_learning -lrunner -lcuda -lcudart  -Wl,-rpath $(GENERATED_CODE_DIR)

batch_learning:
	$(MAKE) -C $(BATCH_LEARNING_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
