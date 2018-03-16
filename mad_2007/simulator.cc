// Standard C++ includes
#include <random>

// GeNN robotics includes
#include "connectors.h"
#include "spike_csv_recorder.h"
#include "timer.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "mad_2007_CODE/definitions.h"

int main()
{
    {
        Timer<> tim("Allocation:");
        allocateMem();
    }
    {
        Timer<> tim("Initialization:");
        initialize();
    }

    {
        Timer<> tim("Building connectivity:");
#ifndef CPU_ONLY
        std::mt19937 rng;
#endif
        buildFixedNumberTotalWithReplacementConnector(Parameters::numInhibitory, Parameters::numInhibitory, Parameters::numIIConnections,
                                                      CII, rng);
        buildFixedNumberTotalWithReplacementConnector(Parameters::numInhibitory, Parameters::numExcitatory, Parameters::numIEConnections,
                                                      CIE, rng);
        buildFixedNumberTotalWithReplacementConnector(Parameters::numExcitatory, Parameters::numExcitatory, Parameters::numEEConnections,
                                                      CEE, rng);
        buildFixedNumberTotalWithReplacementConnector(Parameters::numExcitatory, Parameters::numInhibitory, Parameters::numEIConnections,
                                                      CEI, rng);
    }

    // Final setup
    {
        Timer<> tim("Sparse init:");
        initmad_2007();
    }

    {
        // Open CSV output files
        SpikeCSVRecorderCached spikes("spikes.csv", glbSpkCntE, glbSpkE);

        {
            Timer<> tim("Simulation:");
            // Loop through timesteps
            for(unsigned int i = 0; i < 10000; i++)
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
    }

    return 0;
}
