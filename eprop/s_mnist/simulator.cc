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

// Auto-generated model code
#include "s_mnist_CODE/definitions.h"

// Batch-learning includes
#include "batch_learning.h"

// Model parameters
#include "parameters.h"

//----------------------------------------------------------------------------
// Anonynous namespace
//----------------------------------------------------------------------------
namespace
{
uint32_t readBigEndian(std::ifstream &data)
{
    union
    {
        char b[4];
        uint32_t w;
    } swizzle;

    // Read data into swizzle union
    data.read(&swizzle.b[0], 4);

    // Swap endianess
    std::swap(swizzle.b[0], swizzle.b[3]);
    std::swap(swizzle.b[1], swizzle.b[2]);
    return swizzle.w;
}

unsigned int loadImageData(const std::string &imageDatafilename, uint8_t *&egp,
                           void (*allocateEGPFn)(unsigned int), void (*pushEGPFn)(unsigned int))
{
    // Open binary file
    std::ifstream imageData(imageDatafilename, std::ifstream::binary);
    assert(imageData.good());

    // Read header words
    const uint32_t magic = readBigEndian(imageData);
    const uint32_t numImages = readBigEndian(imageData);
    const uint32_t numRows = readBigEndian(imageData);
    const uint32_t numCols = readBigEndian(imageData);

    // Validate header words
    assert(magic == 0x803);
    assert(numRows == Parameters::inputHeight);
    assert(numCols == Parameters::inputWidth);

    // Allocate EGP for data
    allocateEGPFn(numRows * numCols * numImages);

    // Read data into EGP
    imageData.read(reinterpret_cast<char *>(egp), numRows * numCols * numImages);

    // Push EGP
    pushEGPFn(numRows * numCols * numImages);

    return numImages;
}

void loadLabelData(const std::string &labelDataFilename, unsigned int desiredNumLabels, uint8_t *&egp,
                   void (*allocateEGPFn)(unsigned int), void (*pushEGPFn)(unsigned int))
{
    // Open binary file
    std::ifstream labelData(labelDataFilename, std::ifstream::binary);
    assert(labelData.good());

    // Read header words
    const uint32_t magic = readBigEndian(labelData);
    const uint32_t numLabels = readBigEndian(labelData);

    // Validate header words
    assert(magic == 0x801);
    assert(numLabels == desiredNumLabels);

    // Allocate EGP for data
    allocateEGPFn(numLabels);

    // Read data into EGP
    labelData.read(reinterpret_cast<char *>(egp), numLabels);

    // Push EGP
    pushEGPFn(numLabels);
}

void saveDenseWeights(const std::string &weightFilename, const scalar *weights, unsigned int numPre, unsigned int numPost)
{
    std::ofstream file(weightFilename, std::ifstream::binary);    
    file.write(reinterpret_cast<const char*>(weights), sizeof(scalar) * numPre * numPost);
}
}   // Anonymous namespace

int main()
{
    try
    {
        allocateMem();
#ifdef ENABLE_RECORDING
        allocateRecordingBuffers(Parameters::batchSize * Parameters::trialTimesteps);
#endif
        initialize();

        // Load training data and labels
        const unsigned int numTrainingImages = loadImageData("mnist/train-images.idx3-ubyte", datasetInput, &allocatedatasetInput, &pushdatasetInputToDevice);
        loadLabelData("mnist/train-labels.idx1-ubyte", numTrainingImages, labelsOutput, &allocatelabelsOutput, &pushlabelsOutputToDevice);

        // Allocate indices buffer and initialize host indices
        allocateindicesInput(numTrainingImages);
        allocateindicesOutput(numTrainingImages);
        std::iota(&indicesInput[0], &indicesInput[numTrainingImages], 0);

        // Calculate number of batches this equates to
        const unsigned int numBatches = ((numTrainingImages + Parameters::batchSize - 1) / Parameters::batchSize);

        // Use CUDA to calculate initial transpose of feedforward recurrent->output weights
        BatchLearning::transposeCUDA(d_gRecurrentALIFOutput, d_gOutputRecurrentALIF, 
                                     Parameters::numRecurrentNeurons, Parameters::numOutputNeurons);
        initializeSparse();

        std::ofstream performance("performance.csv");
        performance << "Epoch, Batch, Num trials, Number correct" << std::endl;

        float learningRate = 0.001f;

        // Loop through epochs
        for(unsigned int epoch = 0; epoch < 1; epoch++) {
            std::cout << "Epoch " << epoch << std::endl;

            // Reset GeNN timestep
            t = 0.0f;
            iT = 0;
            
            // Shuffle indices, duplicate to output and upload
            // **TODO** some sort of shared pointer business
            //std::random_shuffle(&indicesInput[0], &indicesInput[numTrainingImages]);
            std::copy_n(indicesInput, numTrainingImages, indicesOutput);
            pushindicesInputToDevice(numTrainingImages);
            pushindicesOutputToDevice(numTrainingImages);

            // Loop through batches in epoch
            unsigned int i = 0;
            for(unsigned int batch = 0; batch < numBatches; batch++) {
                std::cout << "\tBatch " << batch << "/" << numBatches << std::endl;

#ifdef ENABLE_RECORDING
                const std::string filenameSuffix = std::to_string(epoch) + "_" + std::to_string(batch);
                AnalogueRecorder<scalar> outputRecorder("output_" + filenameSuffix + ".csv", {PiOutput, EOutput}, Parameters::numOutputNeurons, ",");
#endif
                // Calculate number of trials in this batch
                const unsigned int numTrialsInBatch = (batch == (numBatches - 1)) ? ((numTrainingImages - 1) % Parameters::batchSize) + 1 : Parameters::batchSize;

                // Loop through trials
                unsigned int numCorrect = 0;
                for(unsigned int trial = 0; trial < numTrialsInBatch; trial++) {
                    // Loop through timesteps
                    std::array<scalar, 10> output{0};
                    for(unsigned int timestep = 0; timestep < Parameters::trialTimesteps; timestep++) {
                        stepTime();

                        // If we're in the cue region
                        if(timestep > (Parameters::inputWidth * Parameters::inputHeight * Parameters::inputRepeats)) {
                            // Download network output
                            pullPiOutputFromDevice();
#ifdef ENABLE_RECORDING
                            pullEOutputFromDevice();

                            // Record outputs
                            outputRecorder.record((double)((Parameters::trialTimesteps * trial) + timestep));
#endif

                            // Add output to total
                            std::transform(output.begin(), output.end(), PiOutput, output.begin(),
                                           [](scalar a, scalar b) { return a + b; });
                        }
                    }

                    // If maximum output matches label, increment counter
                    const auto classification = std::distance(output.cbegin(), std::max_element(output.cbegin(), output.cend()));
                    if(classification == labelsOutput[indicesOutput[i]]) {
                        numCorrect++;
                    }

                    // Advance to next stimuli
                    i++;
                }
#ifdef ENABLE_RECORDING
                pullRecordingBuffersFromDevice();
                writeTextSpikeRecording("input_spikes_" + filenameSuffix + ".csv", recordSpkInput,
                                        Parameters::numInputNeurons, Parameters::batchSize * Parameters::trialTimesteps, Parameters::timestepMs,
                                        ",", true);
                writeTextSpikeRecording("recurrent_alif_spikes_" + filenameSuffix + ".csv", recordSpkRecurrentALIF,
                                        Parameters::numRecurrentNeurons, Parameters::batchSize * Parameters::trialTimesteps, Parameters::timestepMs,
                                        ",", true);
#endif
                // Update weights
                #define ADAM_OPTIMIZER_CUDA(POP_NAME, NUM_SRC_NEURONS, NUM_TRG_NEURONS)   BatchLearning::adamOptimizerCUDA(d_DeltaG##POP_NAME, d_M##POP_NAME, d_V##POP_NAME, d_g##POP_NAME, NUM_SRC_NEURONS, NUM_TRG_NEURONS, epoch, learningRate)

                ADAM_OPTIMIZER_CUDA(InputRecurrentALIF, Parameters::numInputNeurons, Parameters::numRecurrentNeurons);
                ADAM_OPTIMIZER_CUDA(ALIFALIFRecurrent, Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons);

                BatchLearning::adamOptimizerTransposeCUDA(d_DeltaGRecurrentALIFOutput, d_MRecurrentALIFOutput, d_VRecurrentALIFOutput, d_gRecurrentALIFOutput, d_gOutputRecurrentALIF, 
                                                          Parameters::numRecurrentNeurons, Parameters::numOutputNeurons, 
                                                          epoch, learningRate);

                // Update biases
                BatchLearning::adamOptimizerCUDA(d_DeltaBOutput, d_MOutput, d_VOutput, d_BOutput,
                                                 Parameters::numOutputNeurons, 1,
                                                 epoch, learningRate);

                // Display performance in this epoch
                std::cout << "\t\t" << numCorrect << "/" << numTrialsInBatch << "  correct" << std::endl;

                // Write performance to file
                performance << epoch << ", " << batch << ", " << numTrialsInBatch << ", " << numCorrect << std::endl;
            }
            
            // Copy feedforward weights from device
            pullgInputRecurrentALIFFromDevice();
            pullgALIFALIFRecurrentFromDevice();
            pullgRecurrentALIFOutputFromDevice();
            
            // Save to disk
            saveDenseWeights("g_input_recurrent_" + std::to_string(epoch) + ".bin", gInputRecurrentALIF, 
                             Parameters::numInputNeurons, Parameters::numRecurrentNeurons);
            saveDenseWeights("g_recurrent_recurrent_" + std::to_string(epoch) + ".bin", gALIFALIFRecurrent, 
                             Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons);
            saveDenseWeights("g_recurrent_output_" + std::to_string(epoch) + ".bin", gRecurrentALIFOutput, 
                             Parameters::numRecurrentNeurons, Parameters::numOutputNeurons);
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
