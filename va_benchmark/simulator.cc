// Standard C++ includes
#include <iostream>
#include <random>

// GeNN robotics includes
#include "timer.h"
#include "spikeRecorder.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "va_benchmark_CODE/definitions.h"
#include "va_benchmark_CODE/macroLookup.h"
int main()
{
    try
    {
        allocateMem();
        allocateRecordingBuffers(Parameters::numTimesteps);
        initialize();
        initializeSparse();

        {
            Timer a("Simulation wall clock:");
            while(iT < Parameters::numTimesteps) {
                stepTime();
            }
        }
        
        {
            Timer a("Downloading spikes:");
            pullRecordingBuffersFromDevice();
            
            writeTextSpikeRecording("spikes_e.csv", GET_FIELD(E, recordSpk), Parameters::numExcitatory, 
                                    Parameters::numTimesteps, Parameters::timestep, ",", true);
            writeTextSpikeRecording("spikes_i.csv", GET_FIELD(I, recordSpk), Parameters::numInhibitory, 
                                    Parameters::numTimesteps, Parameters::timestep, ",", true);
        }
     
        std::cout << "Init:" << initTime << std::endl;
        std::cout << "Init sparse:" << initSparseTime << std::endl;
        std::cout << "Neuron update:" << neuronUpdateTime << std::endl;
        std::cout << "Presynaptic update:" << presynapticUpdateTime << std::endl;
    }
    catch(const std::exception &ex) {
        std::cerr << "Error:" << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
