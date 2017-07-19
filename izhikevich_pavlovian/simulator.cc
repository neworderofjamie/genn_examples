#include <algorithm>
#include <numeric>
#include <random>

#include "../common/connectors.h"
#include "../common/spike_csv_recorder.h"
#include "../common/timer.h"

#include "izhikevich_pavlovian_CODE/definitions.h"

#include "parameters.h"

int main()
{
    std::mt19937 gen;

    {
        Timer<> t("Allocation:");
        allocateMem();
    }

    {
        Timer<> t("Initialization:");
        initialize();
    }

    {
        Timer<> t("Building connectivity:");
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numInhibitory,
                                    Parameters::probabilityConnection, CII, &allocateII, gen);
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numExcitatory,
                                    Parameters::probabilityConnection, CIE, &allocateIE, gen);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numExcitatory,
                                    Parameters::probabilityConnection, CEE, &allocateEE, gen);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numInhibitory,
                                    Parameters::probabilityConnection, CEI, &allocateEI, gen);
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initizhikevich_pavlovian();
    }

    // Open CSV output files
    SpikeCSVRecorder e_spikes("e_spikes.csv", glbSpkCntE, glbSpkE);
    SpikeCSVRecorder i_spikes("i_spikes.csv", glbSpkCntI, glbSpkI);

    auto simStart = chrono::steady_clock::now();

    // Create distribution to pick an input to apply thamalic input to
    std::uniform_real_distribution<> inputCurrent(-6.5, 6.5);

    {
        Timer<> t("Simulation:");
        // Loop through timesteps
        for(unsigned int t = 0; t < 1000; t++)
        {
            // Generate uniformly distributed numbers to fill host array
            // **TODO** move to GPU
            std::generate_n(IextE, Parameters::numExcitatory,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });
            std::generate_n(IextI, Parameters::numInhibitory,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });

            // Simulate
#ifndef CPU_ONLY
            // Upload random input currents to GPU
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IextE, IextE, Parameters::numExcitatory * sizeof(scalar), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IextI, IextI, Parameters::numInhibitory * sizeof(scalar), cudaMemcpyHostToDevice));

            stepTimeGPU();

            // Download spikes from GPU
            pullECurrentSpikesFromDevice();
            pullICurrentSpikesFromDevice();
#else
            stepTimeCPU();
#endif

            // Record spikes
            e_spikes.record(t);
            i_spikes.record(t);
        }
    }

    return 0;
}
