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

        std::cout << "Initialise host:" << initHost_tme << std::endl;
        std::cout << "Initialise device:" << initDevice_tme << std::endl;
        std::random_device rd;
        std::mt19937 gen(rd());

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_SPARSE
        buildFixedProbabilityConnector(Parameters::numPre, Parameters::numPost, Parameters::connectionProbability,
                                       CSyn, &allocateSyn, gen);
#ifdef SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL
        std::fill(&gSyn[0], &gSyn[CSyn.connN], 0.0f);
#endif  // SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_SPARSE

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_DENSE
#ifdef SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL
        std::fill(&gSyn[0], &gSyn[Parameters::numPre * Parameters::numPost], 0.0f);
#endif  // SYNAPSE_MATRIX_WEIGHT_INDIVIDUAL
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_DENSE

        // Convert input rate into a RNG threshold and fill
        float inputRate = 10E-3f;
        uint64_t baseRates[Parameters::numPre];
        convertRateToRandomNumberThreshold(&inputRate, &baseRates[0], 1);
        std::fill(&baseRates[1], &baseRates[Parameters::numPre], baseRates[0]);

        // Setup reverse connection indices for benchmark
        initbenchmark();
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
