// Standard C++ includes
#include <numeric>
#include <random>

// GeNN robotics includes
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

            float totalWeight = 0.0f;
            unsigned int numSynapses = 0;
            for(unsigned int i = 0; i < 500; i++) {
                for(unsigned int s = 0; s < CIE.rowLength[i]; s++) {
                    totalWeight += gIE[(i * CIE.maxRowLength) + s];
                    numSynapses++;
                }
            }

            // Calculate mean IE weights
            fprintf(weights, "%f, %f\n", 1.0 * (double)t, totalWeight / (double)numSynapses);
        }
    }

    // Close files
    fclose(weights);

    return 0;
}
