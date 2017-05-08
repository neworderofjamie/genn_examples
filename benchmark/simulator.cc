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

        std::random_device rd;
        std::mt19937 gen(rd());

#ifdef SYNAPSE_MATRIX_SPARSE
        buildFixedProbabilityConnector(Parameters::numPre, Parameters::numPost, Parameters::connectionProbability,
                                       CSyn, &allocateSyn, gen);
#ifdef SYNAPSE_MATRIX_INDIVIDUAL
        std::fill(&gSyn[0], &gSyn[CSyn.connN], 0.0f);
#endif  // SYNAPSE_MATRIX_INDIVIDUAL
#else   // !SYNAPSE_MATRIX_SPARSE
        std::fill(&gSyn[0], &gSyn[Parameters::numPre * Parameters::numPost], 0.0f);
#endif  // !SYNAPSE_MATRIX_SPARSE

        // Convert input rate into a RNG threshold and fill
        float inputRate = 10E-3f;
        uint64_t baseRates[Parameters::numPre];
        convertRateToRandomNumberThreshold(&inputRate, &baseRates[0], 1);
        std::fill(&baseRates[1], &baseRates[Parameters::numPre], baseRates[0]);

#ifndef CPU_ONLY
        // Copy base rates to GPU
        uint64_t *d_baseRates = NULL;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_baseRates, sizeof(uint64_t) * Parameters::numPre));
        CHECK_CUDA_ERRORS(cudaMemcpy(d_baseRates, baseRates, sizeof(uint64_t) * Parameters::numPre, cudaMemcpyHostToDevice));
        copyStateToDevice();
        ratesStim = d_baseRates;
#else
        ratesStim = baseRates;
#endif

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
