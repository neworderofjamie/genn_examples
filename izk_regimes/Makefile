GENERATED_CODE_DIR	:=izk_regimes_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: izk_regimes

izk_regimes: simulator.cc generated_code
	$(CXX) $(CXXFLAGS)  simulator.cc -o izk_regimes -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath=$(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
