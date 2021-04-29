// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <vector>

// GeNN userproject includes
#include "spikeRecorder.h"

// Auto-generated model code
#include "deep_unsupervised_learning_inference_CODE/definitions.h"

// GeNN examples includes
#include "../common/mnist_helpers.h"

// Model parameters
#include "parameters.h"

int main()
{
    using namespace Parameters;
    
    allocateMem();
    allocateRecordingBuffers(1000);
    initialize();
    
    // Read kernels from disk
    {
        std::ifstream conv1("kernels/conv1_kernel.bin", std::ios_base::binary);
        std::ifstream conv2("kernels/conv2_kernel.bin", std::ios_base::binary);
        std::ifstream output("kernels/output_kernel.bin", std::ios_base::binary);

        conv1.read(reinterpret_cast<char*>(gInput_Conv1), InputConv1::kernelSize * sizeof(scalar));
        conv2.read(reinterpret_cast<char*>(gConv1_Conv2), Conv1Conv2::kernelSize * sizeof(scalar));
        output.read(reinterpret_cast<char*>(gConv2_Output), Conv2Output::kernelSize * sizeof(scalar));
    }
    initializeSparse();

    // Load training data and labels
    const unsigned int numTrainingImages = loadImageData("train-images-idx3-ubyte", datasetInput, 
                                                         &allocatedatasetInput, &pushdatasetInputToDevice);
    
    // Load labels
    std::vector<uint8_t> labels(numTrainingImages);
    loadLabelData("train-labels-idx1-ubyte", numTrainingImages, labels.data());
    
    std::vector<unsigned int> labelMapping(Output::numNeurons * 10, 0);

    // Loop through training images
    for(unsigned int n = 0; n < numTrainingImages; n++) {
        std::cout << n << std::endl;

        // Simulate
        for(unsigned int i = 0; i < 1000; i++) {
            stepTime();

        }
        pullSpikeCountOutputFromDevice();
        
        // Find most active neuron and increment label mapping
        const size_t mostActive = std::distance(SpikeCountOutput, 
                                                std::max_element(&SpikeCountOutput[0], &SpikeCountOutput[Output::numNeurons]));
        labelMapping[(labels[n] * Output::numNeurons) + mostActive]++;

        if((n % 1000) == 0) {
            // Save spikes
            pullRecordingBuffersFromDevice();
            writeTextSpikeRecording("input_inference_spikes_" + std::to_string(n) + ".csv", recordSpkInput,
                                    Input::numNeurons, 1000, timestepMs,
                                    ",", true);

            writeTextSpikeRecording("conv1_inference_spikes_" + std::to_string(n) + ".csv", recordSpkConv1,
                                    Conv1::numNeurons, 1000, timestepMs,
                                    ",", true);
            
            writeTextSpikeRecording("conv2_inference_spikes_" + std::to_string(n) + ".csv", recordSpkConv2,
                                    Conv2::numNeurons, 1000, timestepMs,
                                    ",", true);
            
            writeTextSpikeRecording("output_inference_spikes_" + std::to_string(n) + ".csv", recordSpkOutput,
                                    Output::numNeurons, 1000, timestepMs,
                                    ",", true);


        }
    }
    
    std::ofstream labelMappingFile("label_mapping.bin", std::ios_base::binary);
    labelMappingFile.write(reinterpret_cast<const char*>(labelMapping.data()), Output::numNeurons * 10 * sizeof(unsigned int));

    return EXIT_SUCCESS;
    
    
}
