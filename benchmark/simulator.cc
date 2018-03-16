#include <random>

#include "connectors.h"
#include "timer.h"

#include "parameters.h"

#include "benchmark_CODE/definitions.h"

int main()
{
    {
        Timer<> t("Allocation:");
        allocateMem();
    }
    {
        Timer<> t("Initialization:");
        initialize();
    }
    std::cout << "\tHost:" << initHost_tme * 1000.0 << std::endl;
    std::cout << "\tDevice:" << initDevice_tme * 1000.0 << std::endl;

    {
        Timer<> t("Building connectivity:");
        std::random_device rd;
        std::mt19937 gen(rd());

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_SPARSE
        buildFixedProbabilityConnector(Parameters::numPre, Parameters::numPost, Parameters::connectionProbability,
                                       CSyn, &allocateSyn, gen);
#elif defined(SYNAPSE_MATRIX_CONNECTIVITY_BITMASK)
        buildFixedProbabilityConnector((Parameters::numPre * Parameters::numPre) / 32 + 1, Parameters::connectionProbability,
                                       gpSyn, gen);
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_SPARSE
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        // Perform sparse initialisation
        initbenchmark();

        // Synchronise to make sure any copy operations are included in the scoped timer
#ifndef CPU_ONLY
        cudaDeviceSynchronize();
#endif
    }
    std::cout << "\tHost:" << sparseInitHost_tme * 1000.0 << std::endl;
    std::cout << "\tDevice:" << sparseInitDevice_tme * 1000.0 << std::endl;

    {
        Timer<> t("Sim:");

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
    std::cout << "\tNeuron:" << neuron_tme * 1000.0 << std::endl;
    std::cout << "\tSynapse:" << synapse_tme * 1000.0 << std::endl;

  return 0;
}
