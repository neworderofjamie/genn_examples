#include <iostream>

#include "parameters.h"

#include "benchmark_CODE/definitions.h"

int main()
{
    allocateMem();
    initialize();
    initializeSparse();


    // Loop through timesteps
    while(iT < 5000) {
        stepTime();
    }

    std::cout << "Timing:" << std::endl;
    std::cout << "\tInit:" << initTime * 1000.0 << std::endl;
    std::cout << "\tSparse init:" << initSparseTime * 1000.0 << std::endl;
    std::cout << "\tNeuron simulation:" << neuronUpdateTime * 1000.0 << std::endl;
    std::cout << "\tPresynaptic update:" << presynapticUpdateTime * 1000.0 << std::endl;
    std::cout << "\tSynapse dynamics:" << synapseDynamicsTime * 1000.0 << std::endl;


  return 0;
}
