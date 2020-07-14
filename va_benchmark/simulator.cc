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
            
            writeBinarySpikeRecording("spikes_e.bin", recordSpkE, 
                                      Parameters::numExcitatory, Parameters::numTimesteps);
            writeBinarySpikeRecording("spikes_i.bin", recordSpkE, 
                                      Parameters::numInhibitory, Parameters::numTimesteps);
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
