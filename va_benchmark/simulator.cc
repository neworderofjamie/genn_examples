// Standard C++ includes
#include <random>

// GeNN robotics includes
#include "connectors.h"
#include "spike_csv_recorder.h"
#include "timer.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "va_benchmark_CODE/definitions.h"

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
        initva_benchmark();
    }

    // Open CSV output files
    SpikeCSVRecorder spikes("spikes.csv", glbSpkCntE, glbSpkE);

    {
        Timer<> t("Simulation:");
        // Loop through timesteps
        for(unsigned int t = 0; t < 10000; t++)
        {
            // Simulate
#ifndef CPU_ONLY
            stepTimeGPU();

            pullECurrentSpikesFromDevice();
#else
            stepTimeCPU();
#endif

            spikes.record(t);
        }
    }

    return 0;
}
