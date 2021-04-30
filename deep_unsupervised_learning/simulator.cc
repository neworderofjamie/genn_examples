// Standard C++ includes
#include <iostream>
#include <random>

// GeNN userproject includes
#include "spikeRecorder.h"

// Auto-generated model code
#include "deep_unsupervised_learning_CODE/definitions.h"


// Model parameters
#include "parameters.h"

int loadImageData(const std::string &imageDatafilename, uint8_t *&egp,
                  void (*allocateEGPFn)(unsigned int), void (*pushEGPFn)(unsigned int))
{
    using namespace Parameters;

    // Open binary file
    std::ifstream imageData(imageDatafilename, std::ifstream::binary);
    assert(imageData.good());

    // Get file length
    imageData.seekg(0, std::ios_base::end);
    const auto fileBytes = imageData.tellg();
    imageData.seekg(0, std::ios_base::beg);

    // Determine how many images this equates to
    const auto numImages = std::div(fileBytes, long{Input::numNeurons});
    assert(numImages.rem == 0);


    // Allocate EGP for data
    allocateEGPFn(Input::numNeurons * numImages.quot);

    // Read data into EGP
    imageData.read(reinterpret_cast<char*>(egp), Input::numNeurons * numImages.quot);

    // Push EGP
    pushEGPFn(Input::numNeurons * numImages.quot);

    return numImages.quot;
}

int main()
{
    using namespace Parameters;
    
    std::mt19937 rng;
    std::normal_distribution<float> conv1WeightDist(0.8f, 0.1f);
    std::normal_distribution<float> conv2WeightDist(0.8f, 0.1f);
    //std::normal_distribution<float> outputWeightDist(0.67f * Conv2Output::poolScale,
    //                                                 0.1f * Conv2Output::poolScale);

    allocateMem();
    allocateRecordingBuffers(1000);
    initialize();

    // Initialise kernels
    // **YUCK** this should be done automatically
    std::generate_n(gInput_Conv1, InputConv1::kernelSize, 
                    [&rng, &conv1WeightDist](){ return conv1WeightDist(rng); });
    std::generate_n(gConv1_Conv2, Conv1Conv2::kernelSize, 
                    [&rng, &conv2WeightDist](){ return conv2WeightDist(rng); });
    //std::generate_n(gConv2_Output, Conv2Output::kernelSize,
    //                [&rng, &outputWeightDist](){ return outputWeightDist(rng); });
    initializeSparse();

    // Load training data and labels
    const unsigned int numTrainingImages = loadImageData("green_grid.bin", datasetInput,
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
                                    Input::numNeurons, 1000, timestepMs,
                                    ",", true);

            writeTextSpikeRecording("conv1_spikes_" + std::to_string(n) + ".csv", recordSpkConv1,
                                    Conv1::numNeurons, 1000, timestepMs,
                                    ",", true);
            writeTextSpikeRecording("conv1_spike_events_" + std::to_string(n) + ".csv", recordSpkEventConv1,
                                    Conv1::numNeurons, 1000, timestepMs,
                                    ",", true);

            writeTextSpikeRecording("conv2_spikes_" + std::to_string(n) + ".csv", recordSpkConv2,
                                    Conv2::numNeurons, 1000, timestepMs,
                                    ",", true);
            //writeTextSpikeRecording("conv2_spike_events_" + std::to_string(n) + ".csv", recordSpkEventConv2,
            //                        Conv2::numNeurons, 1000, timestepMs,
            //                        ",", true);

           // writeTextSpikeRecording("output_spikes_" + std::to_string(n) + ".csv", recordSpkOutput,
            //                        Output::numNeurons, 1000, timestepMs,
            //                        ",", true);
            /*writeTextSpikeRecording("output_spike_events_" + std::to_string(n) + ".csv", recordSpkEventOutput,
                                    1000, 1000, 0.1,
                                    ",", true);*/
            // Save weights
            pullgInput_Conv1FromDevice();
            pullgConv1_Conv2FromDevice();
            //pullgConv2_OutputFromDevice();

            std::ofstream conv1("conv1_kernel_" + std::to_string(n) + ".bin", std::ios_base::binary);
            std::ofstream conv2("conv2_kernel_" + std::to_string(n) + ".bin", std::ios_base::binary);
            //std::ofstream output("output_kernel_" + std::to_string(n) + ".bin", std::ios_base::binary);

            conv1.write(reinterpret_cast<const char*>(gInput_Conv1), InputConv1::kernelSize * sizeof(scalar));
            conv2.write(reinterpret_cast<const char*>(gConv1_Conv2), Conv1Conv2::kernelSize * sizeof(scalar));
            //output.write(reinterpret_cast<const char*>(gConv2_Output), Conv2Output::kernelSize * sizeof(scalar));

        }
    }
    return EXIT_SUCCESS;
    
    
}
