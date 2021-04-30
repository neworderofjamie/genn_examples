GENN_USERPROJECT_INCLUDE	:=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
CXXFLAGS 			+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code generated_code_inference

all: deep_unsupervised_learning deep_unsupervised_learning_inference

deep_unsupervised_learning: simulator.cc generated_code
	$(CXX) $(CXXFLAGS) -I$(GENN_USERPROJECT_INCLUDE) simulator.cc -o deep_unsupervised_learning -Ldeep_unsupervised_learning_CODE -lrunner -Wl,-rpath deep_unsupervised_learning_CODE

deep_unsupervised_learning_inference: simulator_inference.cc generated_code_inference
	$(CXX) $(CXXFLAGS) -I$(GENN_USERPROJECT_INCLUDE) simulator_inference.cc -o deep_unsupervised_learning_inference -Ldeep_unsupervised_learning_inference_CODE -lrunner -Wl,-rpath deep_unsupervised_learning_inference_CODE


generated_code:
	$(MAKE) -C deep_unsupervised_learning_CODE

generated_code_inference:
	$(MAKE) -C deep_unsupervised_learning_inference_CODE
