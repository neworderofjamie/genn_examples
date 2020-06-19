#include <iostream>

#include "parameters.h"
#include "timer.h"

#include "benchmark_CODE/definitions.h"


int main()
{
    try
    {
        allocateMem();
        initialize();
        initializeSparse();

        {
            Timer a("Simulation wall clock:");

            while(t < 10000.0f) {
                stepTime();
            }
        }

        std::cout << "Timing:" << std::endl;
        std::cout << "\tInit:" << initTime << std::endl;
        std::cout << "\tSparse init:" << initSparseTime<< std::endl;
        std::cout << "\tNeuron simulation:" << neuronUpdateTime << std::endl;
        std::cout << "\tSynapse simulation:" << presynapticUpdateTime << std::endl;
    }
    catch(const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
