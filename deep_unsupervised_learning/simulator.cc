// Standard C++ includes
#include <iostream>
#include <random>

// GeNN userproject includes
#include "spikeRecorder.h"

// Auto-generated model code
#include "deep_unsupervised_learning_CODE/definitions.h"

// Model parameters
#include "../common/mnist_helpers.h"

int main()
{
    std::mt19937 rng;
    std::normal_distribution<float> convWeightDist(0.8f, 0.1f);
    
    allocateMem();
    allocateRecordingBuffers(1000);
    initialize();
    
    // Initialise kernels
    // **YUCK** this should be done automatically
    std::generate_n(gInput_Conv1, 5 * 5 * 1 * 16, 
                    [&rng, &convWeightDist](){ return convWeightDist(rng); });
    std::generate_n(gConv1_Conv2, 5 * 5 * 16 * 32, 
                    [&rng, &convWeightDist](){ return convWeightDist(rng); });

    initializeSparse();
    
    // Load training data and labels
    const unsigned int numTrainingImages = loadImageData("train-images-idx3-ubyte", datasetInput, 
                                                         &allocatedatasetInput, &pushdatasetInputToDevice);
    
    // Loop through training images
    for(unsigned int n = 0; n < numTrainingImages; n++) {
        std::cout << n << std::endl;
        
        // Simulate
        for(unsigned int i = 0; i < 1000; i++) {
            stepTime();
        }
        
        if((n % 100) == 0) {
            // Save spikes
            pullRecordingBuffersFromDevice();
            /*writeTextSpikeRecording("input_spikes_" + std::to_string(n) + ".csv", recordSpkInput,
                                    28 * 28, 100, 0.1,
                                    ",", true);*/
            writeTextSpikeRecording("conv1_spikes_" + std::to_string(n) + ".csv", recordSpkConv1,
                                    24 * 24 * 16, 100, 0.1,
                                    ",", true);
            /*writeTextSpikeRecording("output_spikes_" + std::to_string(n) + ".csv", recordSpkOutput,
                                    1000, 100, 0.1,
                                    ",", true);*/
            writeTextSpikeRecording("conv1_spike_events_" + std::to_string(n) + ".csv", recordSpkEventConv1,
                                    24 * 24 * 16, 100, 0.1,
                                    ",", true);
                                    
            // Save weights
            pullgInput_Conv1FromDevice();
            pullgConv1_Conv2FromDevice();
            //pullgInput_OutputFromDevice();
            std::ofstream conv1("conv1_kernel_" + std::to_string(n) + ".bin", std::ios_base::binary);
            std::ofstream conv2("conv2_kernel_" + std::to_string(n) + ".bin", std::ios_base::binary);
            //std::ofstream inputOutput("input_output_" + std::to_string(n) + ".bin", std::ios_base::binary);
            conv1.write(reinterpret_cast<const char*>(gInput_Conv1), 5 * 5 * 1 * 16 * sizeof(float));
            conv2.write(reinterpret_cast<const char*>(gConv1_Conv2), 5 * 5 * 16 * 32 * sizeof(float));
            //inputOutput.write(reinterpret_cast<const char*>(gInput_Output), 1000 * 784 * sizeof(float));
        }
    }
    return EXIT_SUCCESS;
    
    
}
