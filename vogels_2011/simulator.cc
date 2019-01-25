// Standard C++ includes
#include <numeric>
#include <random>

// GeNN robotics includes
#include "common/timer.h"
#include "genn_utils/spike_csv_recorder.h"

// Auto-generated model code
#include "vogels_2011_CODE/definitions.h"

using namespace BoBRobotics;

int main()
{
    {
        Timer<> b("Allocation:");
        allocateMem();
    }
    {
        Timer<> b("Initialization:");
        initialize();
    }

    // Final setup
    {
        Timer<> b("Sparse init:");
        initializeSparse();
    }

    // Download IE connectivity from device
    pullIEConnectivityFromDevice();

    // Open CSV output files
    GeNNUtils::SpikeCSVRecorder spikes("spikes.csv", glbSpkCntE, glbSpkE);

    FILE *weights = fopen("weights.csv", "w");
    fprintf(weights, "Time(ms), Weight (nA)\n");

    {
        Timer<> b("Simulation:");
        // Loop through timesteps
        while(t < 10000.0f) {
            // Simulate
            stepTime();

            pullECurrentSpikesFromDevice();
            pullIEStateFromDevice();

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
