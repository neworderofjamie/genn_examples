#include <algorithm>
#include <chrono>
#include <random>

#include "modelSpec.h"

#include "../common/connectors.h"
#include "../common/timer.h"

#include "parameters.h"

#include "benchmark_CODE/definitions.h"

int main()
{
    {
        Timer<std::milli> t("Alloc:");
        allocateMem();
    }

    {
        Timer<std::milli> t("Init:");
        initialize();

        std::cout << "Initialise host:" << initHost_tme * 1000.0 << std::endl;
        std::cout << "Initialise device:" << initDevice_tme * 1000.0 << std::endl;

        std::random_device rd;
        std::mt19937 gen(rd());

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_SPARSE
        buildFixedProbabilityConnector(Parameters::numPre, Parameters::numPost, Parameters::connectionProbability,
                                       CSyn, &allocateSyn, gen);
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_SPARSE

        // Perform sparse initialisation
        initbenchmark();

        // Synchronise to make sure any copy operations are included in the scoped timer
        cudaDeviceSynchronize();
    }

    {
        Timer<std::milli> t("Sim:");

        // Loop through timesteps
        for(unsigned int t = 0; t < 5000; t++)
        {
            // Simulate
#ifndef CPU_ONLY
            stepTimeGPU();
#else
            stepTimeCPU();
#endif
        }
    }

  return 0;
}
