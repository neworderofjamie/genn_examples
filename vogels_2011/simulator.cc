// Standard C++ includes
#include <numeric>
#include <random>

// GeNN robotics includes
#include "connectors.h"
#include "spike_csv_recorder.h"
#include "timer.h"

// Auto-generated model code
#include "vogels_2011_CODE/definitions.h"

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
        buildFixedProbabilityConnector(500, 500, 0.02f,
                                    CII, rng);
        buildFixedProbabilityConnector(500, 2000, 0.02f,
                                    CIE, rng);
        buildFixedProbabilityConnector(2000, 2000, 0.02f,
                                    CEE, rng);
        buildFixedProbabilityConnector(2000, 500, 0.02f,
                                    CEI, rng);
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initvogels_2011();
    }

    // Open CSV output files
    SpikeCSVRecorder spikes("spikes.csv", glbSpkCntE, glbSpkE);

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
