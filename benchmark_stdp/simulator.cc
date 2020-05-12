#include <iostream>

#include "parameters.h"

#include "benchmark_stdp_CODE/definitions.h"


int main()
{
    allocateMem();
    initialize();
    initializeSparse();

    while(t < 5000.0f) {
        stepTime();
    }

    std::cout << "Timing:" << std::endl;
    std::cout << "\tInit:" << initTime * 1000.0 << std::endl;
    std::cout << "\tSparse init:" << initSparseTime * 1000.0 << std::endl;
    std::cout << "\tNeuron update:" << neuronUpdateTime * 1000.0 << std::endl;
    std::cout << "\tPresynaptic update:" << presynapticUpdateTime * 1000.0 << std::endl;
    std::cout << "\tPostsynaptic update:" << postsynapticUpdateTime * 1000.0 << std::endl;

    return 0;
}
