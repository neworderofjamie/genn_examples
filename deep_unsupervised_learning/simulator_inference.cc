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

bool test()
{
    using namespace Parameters;

    std::ifstream labelFile("neuron_labels.bin", std::ios_base::binary);
    if(!labelFile.good()) {
        return false;
    }

    std::cout << "Testing..." << std::endl;

    // Load neuron labels
    std::vector<unsigned int> neuronLabel(Output::numNeurons);
    labelFile.read(reinterpret_cast<char*>(neuronLabel.data()), Output::numNeurons * sizeof(unsigned int));

    // Load testig data
    const unsigned int numTestingImages = loadImageData("t10k-images-idx3-ubyte", datasetInput,
                                                        &allocatedatasetInput, &pushdatasetInputToDevice);

    // Load testing labels
    std::vector<uint8_t> labels(numTestingImages);
    loadLabelData("t10k-labels-idx1-ubyte", numTestingImages, labels.data());

    // Loop through training images
    unsigned int numCorrect = 0;
    for(unsigned int n = 0; n < numTestingImages; n++) {
        std::cout << n << " (" << int{labels[n]} << ")" << std::endl;

        // Simulate
        for(unsigned int i = 0; i < 1000; i++) {
            stepTime();

        }
        pullSpikeCountOutputFromDevice();

        // Find most active neuron and increment label mapping
        const size_t mostActive = std::distance(&SpikeCountOutput[0],
                                                std::max_element(&SpikeCountOutput[0], &SpikeCountOutput[Output::numNeurons]));

        // Compare to label
        if(neuronLabel[mostActive] == labels[n]) {
            std::cout << "\tCorrect" << std::endl;
            numCorrect++;
        }
        else {
            std::cout << "\tIncorrect (" << neuronLabel[mostActive] << ")" << std::endl;
        }

    }

    // Print performance
    std::cout << numCorrect << "/" << numTestingImages << "(" << ((double)numCorrect / (double)numTestingImages) * 100.0 << "%)" << std::endl;
    return true;
}

void label()
{
    using namespace Parameters;

    std::cout << "Labelling..." << std::endl;

    // Load training data
    const unsigned int numTrainingImages = loadImageData("train-images-idx3-ubyte", datasetInput, 
                                                         &allocatedatasetInput, &pushdatasetInputToDevice);

    // Load training labels
    std::vector<uint8_t> labels(numTrainingImages);
    loadLabelData("train-labels-idx1-ubyte", numTrainingImages, labels.data());

    // Build vector of vectors to hold label counts for each output neuron
    std::vector<std::vector<unsigned int>> labelMapping(Output::numNeurons);
    for(auto &l : labelMapping) {
        l.resize(10, 0);
    }

    // Loop through training images
    for(unsigned int n = 0; n < numTrainingImages; n++) {
        std::cout << n << " (" << int{labels[n]} << ")" << std::endl;

        // Simulate
        for(unsigned int i = 0; i < 1000; i++) {
            stepTime();

        }
        pullSpikeCountOutputFromDevice();

        // Find most active neuron and increment label mapping
        const size_t mostActive = std::distance(&SpikeCountOutput[0],
                                                std::max_element(&SpikeCountOutput[0], &SpikeCountOutput[Output::numNeurons]));
        std::cout << "\tMost active = " << mostActive << std::endl;
        labelMapping[mostActive][labels[n]]++;

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

    // Loop through output neurons
    std::vector<unsigned int> neuronLabel(Output::numNeurons);
    for(unsigned int i = 0; i < Output::numNeurons; i++) {
        // Find label which has been assigned to this neuron the most times
        const auto &neuronLabels = labelMapping[i];
        neuronLabel[i] = std::distance(neuronLabels.cbegin(),
                                       std::max_element(neuronLabels.cbegin(), neuronLabels.cend()));
    }
    std::ofstream labelFile("neuron_labels.bin", std::ios_base::binary);
    labelFile.write(reinterpret_cast<const char*>(neuronLabel.data()), Output::numNeurons * sizeof(unsigned int));

}
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

    // If no labelling data exists for testing, label
    if(!test()) {
        label();
    }

    return EXIT_SUCCESS;
}
