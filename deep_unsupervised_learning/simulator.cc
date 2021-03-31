// Standard C++ includes
#include <iostream>
#include <random>

// GeNN userproject includes
#include "spikeRecorder.h"

// Auto-generated model code
#include "deep_unsupervised_learning_CODE/definitions.h"

// Model parameters
#include "../../common/mnist_helpers.h"

int main()
{
    std::mt19937 rng;
    std::normal_distribution<float> conv2WeightDist(0.8f, 0.1f);
    
    allocateMem();
    allocateRecordingBuffers(100);
    initialize();
    
    // Initialise kernels
    // **YUCK**
    std::generate_n(gInput_Conv1, 5 * 5 * 16, 
                    [&rng, &conv2WeightDist](){ return conv2WeightDist(rng); });
    
    initializeSparse();
    
    // Load training data and labels
    const unsigned int numTrainingImages = loadImageData("train-images-idx3-ubyte", datasetInput, 
                                                         &allocatedatasetInput, &pushdatasetInputToDevice);
    
    // Loop through training images
    for(unsigned int n = 0; n < numTrainingImages; n++) {
        std::cout << n << std::endl;
        
        // Simulate
        for(unsigned int i = 0; i < 100; i++) {
            stepTime();
        }
        
        if((n % 1000) == 0) {
            // Save spikes
            pullRecordingBuffersFromDevice();
            writeTextSpikeRecording("input_spikes_" + std::to_string(n) + ".csv", recordSpkInput,
                                    28 * 28, 100, 1.0,
                                    ",", true);
            writeTextSpikeRecording("conv1_spikes_" + std::to_string(n) + ".csv", recordSpkConv1,
                                    24 * 24 * 16, 100, 1.0,
                                    ",", true);
                                    
            // Save weights
            pullgInput_Conv1FromDevice();
            std::ofstream conv1("conv1_kernel_" + std::to_string(n) + ".bin", std::ios_base::binary);
            conv1.write(reinterpret_cast<const char*>(gInput_Conv1), 5 * 5 * 16 * sizeof(float));
        }
    }
    return EXIT_SUCCESS;
    
    
}