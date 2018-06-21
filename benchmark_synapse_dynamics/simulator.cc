#include <random>

// GeNN robotics includes
#include "common/timer.h"
#include "genn_utils/connectors.h"

#include "parameters.h"

#include "benchmark_CODE/definitions.h"

using namespace BoBRobotics;

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
        GeNNUtils::buildFixedProbabilityConnector(Parameters::numPre, Parameters::numPost, Parameters::connectionProbability,
                                                  CSyn, &allocateSyn, gen);
#elif defined(SYNAPSE_MATRIX_CONNECTIVITY_RAGGED)
        GeNNUtils::buildFixedProbabilityConnector(Parameters::numPre, Parameters::numPost, Parameters::connectionProbability,
                                                  CSyn, gen);
#elif defined(SYNAPSE_MATRIX_CONNECTIVITY_BITMASK)
        GeNNUtils::buildFixedProbabilityConnector((Parameters::numPre * Parameters::numPre) / 32 + 1, Parameters::connectionProbability,
                                                  gpSyn, gen);
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_SPARSE
    }

    // Final setup
    {
        Timer<> t("Sparse init:");

        initbenchmark();
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
    std::cout << "\tSynapse dynamics:" << synDyn_tme * 1000.0 << std::endl;

  return 0;
}
