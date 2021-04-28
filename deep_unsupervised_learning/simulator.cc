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
    std::normal_distribution<float> outputWeightDist(0.67f, 0.1f);

    allocateMem();
    allocateRecordingBuffers(1000);
    initialize();

    // Initialise kernels
    // **YUCK** this should be done automatically
    std::generate_n(gInput_Conv1, 5 * 5 * 1 * 16, 
                    [&rng, &convWeightDist](){ return convWeightDist(rng); });
    std::generate_n(gConv1_Conv2, 5 * 5 * 16 * 32, 
                    [&rng, &convWeightDist](){ return convWeightDist(rng); });
    std::generate_n(gConv2_Output, 32 * 4 * 4 * 1000,
                    [&rng, &outputWeightDist](){ return outputWeightDist(rng); });
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

        if((n % 1000) == 0) {
            // Save spikes
            pullRecordingBuffersFromDevice();
            writeTextSpikeRecording("input_spikes_" + std::to_string(n) + ".csv", recordSpkInput,
                                    28 * 28, 1000, 0.1,
                                    ",", true);

            writeTextSpikeRecording("conv1_spikes_" + std::to_string(n) + ".csv", recordSpkConv1,
                                    24 * 24 * 16, 1000, 0.1,
                                    ",", true);
            writeTextSpikeRecording("conv1_spike_events_" + std::to_string(n) + ".csv", recordSpkEventConv1,
                                    24 * 24 * 16, 1000, 0.1,
                                    ",", true);

            writeTextSpikeRecording("conv2_spikes_" + std::to_string(n) + ".csv", recordSpkConv2,
                                    8 * 8 * 32, 1000, 0.1,
                                    ",", true);
            writeTextSpikeRecording("conv2_spike_events_" + std::to_string(n) + ".csv", recordSpkEventConv2,
                                    8 * 8 * 32, 1000, 0.1,
                                    ",", true);

            writeTextSpikeRecording("output_spikes_" + std::to_string(n) + ".csv", recordSpkOutput,
                                    1000, 1000, 0.1,
                                    ",", true);
            /*writeTextSpikeRecording("output_spike_events_" + std::to_string(n) + ".csv", recordSpkEventOutput,
                                    1000, 1000, 0.1,
                                    ",", true);*/
            // Save weights
            pullgInput_Conv1FromDevice();
            pullgConv1_Conv2FromDevice();
            pullgConv2_OutputFromDevice();

            std::ofstream conv1("conv1_kernel_" + std::to_string(n) + ".bin", std::ios_base::binary);
            std::ofstream conv2("conv2_kernel_" + std::to_string(n) + ".bin", std::ios_base::binary);
            std::ofstream output("output_kernel_" + std::to_string(n) + ".bin", std::ios_base::binary);

            conv1.write(reinterpret_cast<const char*>(gInput_Conv1), 5 * 5 * 1 * 16 * sizeof(float));
            conv2.write(reinterpret_cast<const char*>(gConv1_Conv2), 5 * 5 * 16 * 32 * sizeof(float));
            output.write(reinterpret_cast<const char*>(gConv2_Output), 32 * 4 * 4 * 1000 * sizeof(float));

        }
    }
    return EXIT_SUCCESS;
    
    
}
