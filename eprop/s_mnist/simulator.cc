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
#include "s_mnist_CODE/definitions.h"

// Model parameters
#include "../../common/mnist_helpers.h"
#include "parameters.h"

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
        const unsigned int numTrainingImages = loadImageData("mnist/train-images-idx3-ubyte", datasetInput, 
                                                             &allocatedatasetInput, &pushdatasetInputToDevice);
        loadLabelData("mnist/train-labels-idx1-ubyte", numTrainingImages, labelsOutput, 
                      &allocatelabelsOutput, &pushlabelsOutputToDevice);

        // Allocate indices buffer and initialize host indices
        allocateindicesInput(numTrainingImages);
        allocateindicesOutput(numTrainingImages);
        std::iota(&indicesInput[0], &indicesInput[numTrainingImages], 0);

        // Calculate number of batches this equates to
        const unsigned int numBatches = ((numTrainingImages + Parameters::batchSize - 1) / Parameters::batchSize);

#ifdef RESUME_EPOCH
        // Load from disk
        loadDense("g_input_recurrent_" + std::to_string(RESUME_EPOCH) + ".bin", gInputRecurrentALIF, 
                  Parameters::numInputNeurons * Parameters::numRecurrentNeurons);
        loadDense("g_recurrent_recurrent_" + std::to_string(RESUME_EPOCH) + ".bin", gALIFALIFRecurrent, 
                  Parameters::numRecurrentNeurons * Parameters::numRecurrentNeurons);
        loadDense("g_recurrent_output_" + std::to_string(RESUME_EPOCH) + ".bin", gRecurrentALIFOutput, 
                  Parameters::numRecurrentNeurons * Parameters::numOutputNeurons);
        loadDense("b_output_" + std::to_string(RESUME_EPOCH) + ".bin", BOutput,
                  Parameters::numOutputNeurons);
        const unsigned int startEpoch = RESUME_EPOCH + 1;
#else
        const unsigned int startEpoch = 0;
#endif
        initializeSparse();
        
        // Calculate initial transpose
        updateCalculateTranspose();

        std::ofstream performance("performance.csv");
        performance << "Epoch, Batch, Num trials, Number correct" << std::endl;

        float learningRate = 0.001f;

        // Loop through epochs
        for(unsigned int epoch = startEpoch; epoch < 10; epoch++) {
            std::cout << "Epoch " << epoch << std::endl;

            // Reset GeNN timestep
            t = 0.0f;
            iT = 0;
            
            // Shuffle indices, duplicate to output and upload
            // **TODO** some sort of shared pointer business
            std::random_shuffle(&indicesInput[0], &indicesInput[numTrainingImages]);
            std::copy_n(indicesInput, numTrainingImages, indicesOutput);
            pushindicesInputToDevice(numTrainingImages);
            pushindicesOutputToDevice(numTrainingImages);

            // Loop through batches in epoch
            unsigned int i = 0;
            for(unsigned int batch = 0; batch < numBatches; batch++) {
                Timer batchTimer("\t\tTime: ");
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
                const unsigned int adamStep = (epoch * numBatches) + batch;
                
                // Apply learning
                const scalar firstMomentScale = 1.0f / (1.0f - std::pow(Parameters::adamBeta1, adamStep + 1));
                const scalar secondMomentScale = 1.0f / (1.0f - std::pow(Parameters::adamBeta2, adamStep + 1));
                alphaInputRecurrentWeightOptimiser = learningRate;
                alphaRecurrentRecurrentWeightOptimiser = learningRate;
                alphaRecurrentOutputWeightOptimiser = learningRate;
                firstMomentScaleInputRecurrentWeightOptimiser = firstMomentScale;
                firstMomentScaleRecurrentRecurrentWeightOptimiser = firstMomentScale;
                firstMomentScaleRecurrentOutputWeightOptimiser = firstMomentScale;
                secondMomentScaleInputRecurrentWeightOptimiser = secondMomentScale;
                secondMomentScaleRecurrentRecurrentWeightOptimiser = secondMomentScale;
                secondMomentScaleRecurrentOutputWeightOptimiser = secondMomentScale;
                updateGradientLearn();

                // Display performance in this epoch
                std::cout << "\t\t" << numCorrect << "/" << numTrialsInBatch << "  correct" << std::endl;

                // Write performance to file
                performance << epoch << ", " << batch << ", " << numTrialsInBatch << ", " << numCorrect << std::endl;
            }

            // Copy feedforward weights and biases from device
            pullgInputRecurrentALIFFromDevice();
            pullgALIFALIFRecurrentFromDevice();
            pullgRecurrentALIFOutputFromDevice();
            pullBOutputFromDevice();

            // Save to disk
            saveDense("g_input_recurrent_" + std::to_string(epoch) + ".bin", gInputRecurrentALIF, 
                      Parameters::numInputNeurons * Parameters::numRecurrentNeurons);
            saveDense("g_recurrent_recurrent_" + std::to_string(epoch) + ".bin", gALIFALIFRecurrent, 
                      Parameters::numRecurrentNeurons * Parameters::numRecurrentNeurons);
            saveDense("g_recurrent_output_" + std::to_string(epoch) + ".bin", gRecurrentALIFOutput, 
                      Parameters::numRecurrentNeurons * Parameters::numOutputNeurons);
            saveDense("b_output_" + std::to_string(epoch) + ".bin", BOutput,
                      Parameters::numOutputNeurons);
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
