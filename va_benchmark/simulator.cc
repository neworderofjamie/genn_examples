// Standard C++ includes
#include <random>

// GeNN robotics includes
#include "common/timer.h"
#include "genn_utils/spike_csv_recorder.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "va_benchmark_CODE/definitions.h"

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

    // Final setup
    {
        Timer<> t("Sparse init:");
        initializeSparse();
    }

    // Open CSV output files
    GeNNUtils::SpikeCSVRecorder spikes("spikes.csv", glbSpkCntE, glbSpkE);

    {
        Timer<> t("Simulation:");
        // Loop through timesteps
        for(unsigned int t = 0; t < 10000; t++)
        {
            // Simulate
            stepTime();

            pullECurrentSpikesFromDevice();


            spikes.record(t);
        }
    }

    return 0;
}
