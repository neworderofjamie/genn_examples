#include <iostream>

#include "parameters.h"

#include "benchmark_CODE/definitions.h"


int main()
{
    allocateMem();
    initialize();
    initializeSparse();

    unsigned int totalSpikes = 0;
    while(t < 5000.0f) {
        stepTime();

        pullNeuronsCurrentSpikesFromDevice();
        totalSpikes += spikeCount_Neurons;
    }

    std::cout << "Timing:" << std::endl;
    std::cout << "\tInit:" << initTime * 1000.0 << std::endl;
    std::cout << "\tSparse init:" << initSparseTime * 1000.0 << std::endl;
    std::cout << "\tNeuron simulation:" << neuronUpdateTime * 1000.0 << std::endl;
    std::cout << "\tSynapse simulation:" << presynapticUpdateTime * 1000.0 << std::endl;

    std::cout << totalSpikes << " spikes emitted" << std::endl;
    return 0;
}
