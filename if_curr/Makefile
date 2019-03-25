GENERATED_CODE_DIR	:=if_curr_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: if_curr

if_curr: simulator.cc generated_code
	$(CXX) $(CXXFLAGS)  simulator.cc -o if_curr -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath=$(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
