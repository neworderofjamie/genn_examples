// Standard C++ includes
#include <numeric>
#include <random>

// GeNN robotics includes
#include "common/timer.h"
#include "genn_utils/connectors.h"
#include "genn_utils/spike_csv_recorder.h"

// Auto-generated model code
#include "vogels_2011_CODE/definitions.h"

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

    {
        Timer<> t("Building connectivity:");
#ifndef CPU_ONLY
        std::mt19937 rng;
#endif
        GeNNUtils::buildFixedProbabilityConnector(500, 500, 0.02f,
                                                  CII, &allocateII, rng);
        GeNNUtils::buildFixedProbabilityConnector(500, 2000, 0.02f,
                                                  CIE, &allocateIE, rng);
        GeNNUtils::buildFixedProbabilityConnector(2000, 2000, 0.02f,
                                                  CEE, &allocateEE, rng);
        GeNNUtils::buildFixedProbabilityConnector(2000, 500, 0.02f,
                                                  CEI, &allocateEI, rng);
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initvogels_2011();
    }

    // Open CSV output files
    GeNNUtils::SpikeCSVRecorder spikes("spikes.csv", glbSpkCntE, glbSpkE);

    FILE *weights = fopen("weights.csv", "w");
    fprintf(weights, "Time(ms), Weight (nA)\n");

    {
        Timer<> t("Simulation:");
        // Loop through timesteps
        for(unsigned int t = 0; t < 10000; t++)
        {
            // Simulate
#ifndef CPU_ONLY
            stepTimeGPU();

            pullECurrentSpikesFromDevice();
            //pullIEStateFromDevice();
#else
            stepTimeCPU();
#endif

            spikes.record(t);


            // Calculate mean IE weights
            float totalWeight = std::accumulate(&gIE[0], &gIE[CIE.connN], 0.0f);
            fprintf(weights, "%f, %f\n", 1.0 * (double)t, totalWeight / (double)CIE.connN);

        }
    }

    // Close files
    fclose(weights);

    return 0;
}
