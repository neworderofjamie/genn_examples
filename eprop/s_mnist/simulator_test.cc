// Standard C++ includes
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

// GeNN userproject includes
#include "analogueRecorder.h"
#include "spikeRecorder.h"
#include "timer.h"

// Auto-generated model code
#include "s_mnist_test_CODE/definitions.h"

// Batch-learning includes
#include "batch_learning.h"

// Model parameters
#include "mnist_helpers.h"
#include "parameters.h"

int main()
{
    try
    {
        allocateMem();
        initialize();

        // Load testing data and labels
        const unsigned int numTestingImages = loadImageData("mnist/t10k-images-idx3-ubyte", datasetInput,
                                                            &allocatedatasetInput, &pushdatasetInputToDevice);

        std::vector<uint8_t> testingLabels(numTestingImages);
        loadLabelData("mnist/t10k-labels-idx1-ubyte", numTestingImages, testingLabels.data());

#ifdef ENABLE_RECORDING
        allocateRecordingBuffers(numTestingImages * Parameters::trialTimesteps);
#endif

        // Allocate indices buffer and initialize host indices
        allocateindicesInput(numTestingImages);
        std::iota(&indicesInput[0], &indicesInput[numTestingImages], 0);
        pushindicesInputToDevice(numTestingImages);

        // Load from disk
        const unsigned int loadEpoch = 1;
        loadDense("g_input_recurrent_" + std::to_string(loadEpoch) + ".bin", gInputRecurrentALIF,
                    Parameters::numInputNeurons * Parameters::numRecurrentNeurons);
        loadDense("g_recurrent_recurrent_" + std::to_string(loadEpoch) + ".bin", gALIFALIFRecurrent,
                    Parameters::numRecurrentNeurons * Parameters::numRecurrentNeurons);
        loadDense("g_recurrent_output_" + std::to_string(loadEpoch) + ".bin", gRecurrentALIFOutput,
                    Parameters::numRecurrentNeurons * Parameters::numOutputNeurons);
        loadDense("b_output_" + std::to_string(loadEpoch) + ".bin", BOutput,
                    Parameters::numOutputNeurons);

        initializeSparse();

#ifdef ENABLE_RECORDING
        AnalogueRecorder<scalar> outputRecorder("test_output.csv", {PiOutput}, Parameters::numOutputNeurons, ",");
#endif

        // Loop through images
        unsigned int numCorrect = 0;
        for(unsigned int image = 0; image < numTestingImages; image++) {
            if((image % 100) == 0) {
                std::cout << "Image " << image << "/" << numTestingImages << std::endl;
            }

            // Loop through timesteps
            std::array<scalar, 10> output{0};
            for(unsigned int timestep = 0; timestep < Parameters::trialTimesteps; timestep++) {
                stepTime();

                // If we're in the cue region
                if(timestep > (Parameters::inputWidth * Parameters::inputHeight * Parameters::inputRepeats)) {
                    // Download network output
                    pullPiOutputFromDevice();

#ifdef ENABLE_RECORDING
                    // Record outputs
                    outputRecorder.record(t);
#endif

                    // Add output to total
                    std::transform(output.begin(), output.end(), PiOutput, output.begin(),
                                   [](scalar a, scalar b) { return a + b; });
                }
            }

            // If maximum output matches label, increment counter
            const auto classification = std::distance(output.cbegin(), std::max_element(output.cbegin(), output.cend()));
            if(classification == testingLabels[image]) {
                numCorrect++;
            }

            if(image != 0 && ((image % 100) == 0)) {
                std::cout << "\t" << ((double)numCorrect / (double)image) * 100.0 << "% accuracy" << std::endl;
            }

        }

#ifdef ENABLE_RECORDING
        pullRecordingBuffersFromDevice();
        writeTextSpikeRecording("test_input_spikes.csv", recordSpkInput,
                                Parameters::numInputNeurons, numTestingImages * Parameters::trialTimesteps, Parameters::timestepMs,
                                ",", true);
        writeTextSpikeRecording("test_recurrent_alif_spikes.csv", recordSpkRecurrentALIF,
                                Parameters::numRecurrentNeurons, numTestingImages * Parameters::trialTimesteps, Parameters::timestepMs,
                                ",", true);
#endif

        // Display performance
        std::cout << numCorrect << "/" << numTestingImages << "  correct" << std::endl;
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
