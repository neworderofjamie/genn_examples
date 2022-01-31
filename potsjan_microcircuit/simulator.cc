// Standard C++ includes
#include <iostream>
#include <memory>
#include <random>
#include <vector>

// GeNN user projects includes
#include "timer.h"
#include "spikeRecorder.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "potjans_microcircuit_CODE/definitions.h"
#include "potjans_microcircuit_CODE/macroLookup.h"

// Macro to record a population's output
#define RECORD_SPIKES(LAYER, POPULATION) \
    writeTextSpikeRecording(#LAYER#POPULATION".csv", GET_FIELD(LAYER##POPULATION,recordSpk), Parameters::getScaledNumNeurons(Parameters::Layer##LAYER, Parameters::Population##POPULATION), timesteps, Parameters::dtMs, ",", true);

int main()
{
    try
    {
        const unsigned int timesteps = (unsigned int)round(Parameters::durationMs / Parameters::dtMs);
        
        allocateMem();
        allocateRecordingBuffers(timesteps);
        initialize();
        initializeSparse();
        double recordS = 0.0;

        {
            Timer timer("Simulation:");
            // Loop through timesteps
            
            const unsigned int tenPercentTimestep = timesteps / 10;
            for(unsigned int i = 0; i < timesteps; i++)
            {
                // Indicate every 10%
                if((i % tenPercentTimestep) == 0) {
                    std::cout << i / 100 << "%" << std::endl;
                }

                // Simulate
                stepTime();
            }
        }

        // Write spike recorder cache to disk
        {
            Timer timer("Recording spikes:");
            pullRecordingBuffersFromDevice();
            RECORD_SPIKES(23, E);
            RECORD_SPIKES(23, I);
            RECORD_SPIKES(4, E);
            RECORD_SPIKES(4, I);
            RECORD_SPIKES(5, E);
            RECORD_SPIKES(5, I);
            RECORD_SPIKES(6, E);
            RECORD_SPIKES(6, I);
        }

        if(Parameters::measureTiming) {
            std::cout << "Timing:" << std::endl;
            std::cout << "\tInit:" << initTime * 1000.0 << std::endl;
            std::cout << "\tSparse init:" << initSparseTime * 1000.0 << std::endl;
            std::cout << "\tNeuron simulation:" << neuronUpdateTime * 1000.0 << std::endl;
            std::cout << "\tSynapse simulation:" << presynapticUpdateTime * 1000.0 << std::endl;
        }
        std::cout << "Record:" << recordS << "ms" << std::endl;
    }
    catch(const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
