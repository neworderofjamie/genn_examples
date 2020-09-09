#include <iostream>

#include "parameters.h"

#include "benchmark_CODE/definitions.h"


int main()
{
    allocateMem();
    initialize();
    initializeSparse();

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_DENSE
    allocatekernelsgSyn(3 * 3);
#endif


    while(t < 5000.0f) {
        stepTime();
    }

    std::cout << "Timing:" << std::endl;
    std::cout << "\tInit:" << initTime * 1000.0 << std::endl;
    std::cout << "\tSparse init:" << initSparseTime * 1000.0 << std::endl;
    std::cout << "\tNeuron simulation:" << neuronUpdateTime * 1000.0 << std::endl;
    std::cout << "\tSynapse simulation:" << presynapticUpdateTime * 1000.0 << std::endl;

    return 0;
}
