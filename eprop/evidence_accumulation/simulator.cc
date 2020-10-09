// Standard C++ includes
#include <iostream>
#include <random>

// GeNN userproject includes
#include "analogueRecorder.h"
#include "spikeRecorder.h"

// Auto-generated model code
#include "evidence_accumulation_CODE/definitions.h"

// Batch-learning includes
#include "batch_learning.h"

// Model parameters
#include "parameters.h"

namespace
{
void setInputRates(float left = Parameters::inactiveRateHz, float right = Parameters::inactiveRateHz, 
                   float decision = Parameters::inactiveRateHz, float background = Parameters::backgroundRateHz)
{
    // Fill rates
    std::fill_n(&rateInput[0 * Parameters::numInputPopulationNeurons], Parameters::numInputPopulationNeurons, left);
    std::fill_n(&rateInput[1 * Parameters::numInputPopulationNeurons], Parameters::numInputPopulationNeurons, right);
    std::fill_n(&rateInput[2 * Parameters::numInputPopulationNeurons], Parameters::numInputPopulationNeurons, decision);
    std::fill_n(&rateInput[3 * Parameters::numInputPopulationNeurons], Parameters::numInputPopulationNeurons, background);

    // Upload to GPU
    pushrateInputToDevice();
}
}

int main()
{
    try
    {
        allocateMem();
        initialize();
        initializeSparse();

        SpikeRecorder<SpikeWriterTextCached> inputSpikeRecorder(&getInputCurrentSpikes, &getInputCurrentSpikeCount, "input_spikes.csv", ",", true);
        SpikeRecorder<SpikeWriterTextCached> recurrentLIFSpikeRecorder(&getRecurrentLIFCurrentSpikes, &getRecurrentLIFCurrentSpikeCount, "recurrent_lif_spikes.csv", ",", true);
        SpikeRecorder<SpikeWriterTextCached> recurrentALIFSpikeRecorder(&getRecurrentALIFCurrentSpikes, &getRecurrentALIFCurrentSpikeCount, "recurrent_alif_spikes.csv", ",", true);
        AnalogueRecorder<float> outputRecorder("output.csv", {YOutput, YStarOutput}, Parameters::numOutputNeurons, ",");

        std::mt19937 rng;
        std::uniform_int_distribution<unsigned int> delayTimestepsDistribution(Parameters::minDelayTimesteps, Parameters::maxDelayTimesteps);
    
        const std::mt19937::result_type midRNG = std::mt19937::min() + ((std::mt19937::max() - std::mt19937::min()) / 2);

        float learningRate = 0.005f;

        // Start with a single cue
        unsigned int numCues = 1;
        for(unsigned int epoch = 0;; epoch++) {
            std::cout << "Epoch " << epoch << std::endl;

            // Loop through trials
            for(unsigned int trial = 0; trial < 64; trial++) {
                std::cout << "Trial " << trial << std::endl;
                // Calculate number of timesteps per cue in this trial
                const unsigned int cueTimesteps = (Parameters::cuePresentTimesteps + Parameters::cueDelayTimesteps) * numCues;

                // Pick delay time
                const unsigned int delayTimesteps = delayTimestepsDistribution(rng);

                // Loop through trial timesteps
                unsigned int numLeftCues = 0;
                unsigned int numRightCues = 0;
                const unsigned int trialTimesteps = cueTimesteps + delayTimesteps + Parameters::decisionTimesteps;
                for(unsigned int timestep = 0; timestep < trialTimesteps; timestep++) {
                    // Cue
                    if(timestep < cueTimesteps) {
                        // Figure out what timestep within the cue we're in
                        const unsigned int cueTimestep = timestep % (Parameters::cuePresentTimesteps + Parameters::cueDelayTimesteps);
                    
                        // If this is the first timestep of the cue
                        if(cueTimestep == 0) {
                            // Activate either left or right neuron
                            if(rng() < midRNG) {
                                numLeftCues++;
                                setInputRates(Parameters::activeRateHz, Parameters::inactiveRateHz);
                            }
                            else {
                                numRightCues++;
                                setInputRates(Parameters::inactiveRateHz, Parameters::activeRateHz);
                            }
                        }
                        // Otherwise, if this is the last timestep of the cue
                        else if(cueTimestep == Parameters::cuePresentTimesteps) {
                            setInputRates();
                        }
                    }
                    // Delay
                    else if(timestep == cueTimesteps) {
                        setInputRates();
                    }
                    // Decision
                    else if(timestep == (cueTimesteps + delayTimesteps)){
                        // Activate correct output neuron depending on which cue was presented more often
                        if(numLeftCues > numRightCues) {
                            YStarOutput[0] = 1.0f;  YStarOutput[1] = 0.0f;
                        }
                        else {
                            YStarOutput[0] = 0.0f;  YStarOutput[1] = 1.0f;
                        }
                        pushYStarOutputToDevice();
                        setInputRates(Parameters::inactiveRateHz, Parameters::inactiveRateHz, Parameters::activeRateHz);
                    }
                    stepTime();

                    // Record spikes
                    // **NOTE** irregular trial lengths make it tricky to use spike recording
                    pullInputCurrentSpikesFromDevice();
                    pullRecurrentLIFCurrentSpikesFromDevice();
                    pullRecurrentALIFCurrentSpikesFromDevice();
                    pullYStarOutputFromDevice();
                    pullYOutputFromDevice();
                    inputSpikeRecorder.record(t);
                    recurrentLIFSpikeRecorder.record(t);
                    recurrentALIFSpikeRecorder.record(t);
                    outputRecorder.record(t);
                }
            }

            // Turn off both outputs
            YStarOutput[0] = 0.0f;  YStarOutput[1] = 0.0f;
            pushYStarOutputToDevice();

            // Apply learning
            BatchLearning::adamOptimizerCUDA(d_DeltaGInputRecurrentLIF, d_MInputRecurrentLIF, d_VInputRecurrentLIF, d_gInputRecurrentLIF,
                                             Parameters::numInputNeurons, Parameters::numRecurrentNeurons,
                                             epoch, learningRate);
            BatchLearning::adamOptimizerCUDA(d_DeltaGInputRecurrentALIF, d_MInputRecurrentALIF, d_VInputRecurrentALIF, d_gInputRecurrentALIF,
                                             Parameters::numInputNeurons, Parameters::numRecurrentNeurons,
                                             epoch, learningRate);
            BatchLearning::adamOptimizerCUDA(d_DeltaGLIFLIFRecurrent, d_MLIFLIFRecurrent, d_VLIFLIFRecurrent, d_gLIFLIFRecurrent,
                                             Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons,
                                             epoch, learningRate);
            BatchLearning::adamOptimizerCUDA(d_DeltaGALIFLIFRecurrent, d_MALIFLIFRecurrent, d_VALIFLIFRecurrent, d_gALIFLIFRecurrent,
                                             Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons,
                                             epoch, learningRate);
            BatchLearning::adamOptimizerCUDA(d_DeltaGLIFALIFRecurrent, d_MLIFALIFRecurrent, d_VLIFALIFRecurrent, d_gLIFALIFRecurrent,
                                             Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons,
                                             epoch, learningRate);
            BatchLearning::adamOptimizerCUDA(d_DeltaGALIFALIFRecurrent, d_MALIFALIFRecurrent, d_VALIFALIFRecurrent, d_gALIFALIFRecurrent,
                                             Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons,
                                             epoch, learningRate);
            BatchLearning::adamOptimizerCUDA(d_DeltaGRecurrentLIFOutput, d_MRecurrentLIFOutput, d_VRecurrentLIFOutput, d_gRecurrentLIFOutput,
                                             Parameters::numRecurrentNeurons, Parameters::numOutputNeurons,
                                             epoch, learningRate);
            BatchLearning::adamOptimizerCUDA(d_DeltaGRecurrentALIFOutput, d_MRecurrentALIFOutput, d_VRecurrentALIFOutput, d_gRecurrentALIFOutput,
                                             Parameters::numRecurrentNeurons, Parameters::numOutputNeurons,
                                             epoch, learningRate);
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}