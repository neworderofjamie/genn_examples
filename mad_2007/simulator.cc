// Standard C++ includes
#include <fstream>
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
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                       gpII, rng);
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                       gpIE, rng);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                       gpEI, rng);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                       CEE, rng);
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
    /*{
        Timer<> tim("Weight analysis:");

        // Download weights
        pullEEStateFromDevice();

        // Write row weights to file
        std::ofstream weights("weights.bin", std::ios::binary);
        for(unsigned int i = 0; i < Parameters::numInhibitory; i++) {
            weights.write(reinterpret_cast<char*>(&gEE[i * CEE.maxRowLength]), sizeof(scalar) * CEE.rowLength[i]);
        }
    }*/

    return 0;
}
