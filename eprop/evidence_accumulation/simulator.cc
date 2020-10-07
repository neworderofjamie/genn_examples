// Standard C++ includes
#include <iostream>
#include <random>

// GeNN userproject includes
#include "spikeRecorder.h"

// Auto-generated model code
#include "evidence_accumulation_CODE/definitions.h"

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

        std::mt19937 rng;
        std::uniform_int_distribution<unsigned int> delayTimestepsDistribution(Parameters::minDelayTimesteps, Parameters::maxDelayTimesteps);
    
        const std::mt19937::result_type midRNG = std::mt19937::min() + ((std::mt19937::max() - std::mt19937::min()) / 2);
        // Start with a single cue
        unsigned int numCues = 1;
        while(true) {
            for(unsigned int trial = 0; trial < 64; trial++) {
                std::cout << "Trial " << trial << std::endl;
                // Calculate number of timesteps per cue in this trial
                const unsigned int cueTimesteps = (Parameters::cuePresentTimesteps + Parameters::cueDelayTimesteps) * numCues;

                // Pick delay time
                const unsigned int delayTimesteps = delayTimestepsDistribution(rng);

                // Loop through trial timesteps
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
                                setInputRates(Parameters::activeRateHz, Parameters::inactiveRateHz);
                            }
                            else {
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
                        setInputRates(Parameters::inactiveRateHz, Parameters::inactiveRateHz, Parameters::activeRateHz);
                    }
                    stepTime();

                    // Record spikes
                    // **NOTE** irregular trial lengths make it tricky to use spike recording
                    pullInputCurrentSpikesFromDevice();
                    inputSpikeRecorder.record(t);
                }
            }

            // **TODO** weight update

            // Stop 
            break;
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}