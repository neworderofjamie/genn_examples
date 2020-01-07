// Standard C++ includes
#include <numeric>
#include <random>

// GeNN userproject includes
#include "timer.h"
#include "spikeRecorder.h"

// Auto-generated model code
#include "vogels_2011_CODE/definitions.h"

int main()
{
    allocateMem();
    initialize();
    initializeSparse();

    // Download IE connectivity from device
    pullIEConnectivityFromDevice();

    // Open CSV output files
    SpikeRecorder<SpikeWriterTextCached> spikes(&getECurrentSpikes, &getECurrentSpikeCount, "spikes.csv", ",", true);

    FILE *weights = fopen("weights.csv", "w");
    fprintf(weights, "Time(ms), Weight (nA)\n");

    {
        Timer b("Simulation:");
        // Loop through timesteps
        while(t < 10000.0f) {
            // Simulate
            stepTime();

            pullECurrentSpikesFromDevice();
            pullgIEFromDevice();

            spikes.record(t);

            float totalWeight = 0.0f;
            unsigned int numSynapses = 0;
            for(unsigned int i = 0; i < 500; i++) {
                for(unsigned int s = 0; s < rowLengthIE[i]; s++) {
                    totalWeight += gIE[(i * maxRowLengthIE) + s];
                    numSynapses++;
                }
            }

            // Calculate mean IE weights
            fprintf(weights, "%f, %f\n", 1.0 * t, totalWeight / (double)numSynapses);
        }
    }

    // Close files
    fclose(weights);

    return 0;
}
