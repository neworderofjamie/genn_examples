GENERATED_CODE_DIR	:=izhikevich_pavlovian_CODE
CXXFLAGS 		+=-std=c++11 -Wall -Wpedantic -Wextra

.PHONY: all clean generated_code

all: izhikevich_pavlovian

izhikevich_pavlovian: simulator.cc generated_code
	$(CXX) $(CXXFLAGS)  simulator.cc -o izhikevich_pavlovian -I$(BOB_ROBOTICS_PATH)/include -L$(GENERATED_CODE_DIR) -lrunner -Wl,-rpath $(GENERATED_CODE_DIR)

generated_code:
	$(MAKE) -C $(GENERATED_CODE_DIR)
