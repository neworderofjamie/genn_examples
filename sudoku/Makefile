GENERATED_CODE_DIR		:=sudoku_CODE
GENN_USERPROJECT_INCLUDE	:=$(abspath $(dir $(shell which genn-buildmodel.sh))../userproject/include)
CXXFLAGS 			+=-std=c++11 -Wall -Wpedantic -Wextra `pkg-config opencv --cflags` -pthread

.PHONY: all clean generated_code

all: sudoku

sudoku: simulator.cc generated_code
	$(CXX) $(CXXFLAGS) -I$(GENN_USERPROJECT_INCLUDE) simulator.cc -o sudoku -ldl `pkg-config opencv --libs`

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
