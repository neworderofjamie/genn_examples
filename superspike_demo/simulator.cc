// Standard C++ includes
#include <iostream>
#include <random>

// GeNN userproject includes
#include "timer.h"
#include "spikeRecorder.h"

// Batch learning includes
#include "batch_learning.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "superspike_demo_CODE/definitions.h"

int main()
{
    try
    {
        allocateMem();
        initialize();

        // Use CUDA to calculate initial transpose of feedforward hidden->output weights
        BatchLearning::transposeCUDA(d_wHidden_Output, d_wOutput_Hidden,
                                    Parameters::numHidden, Parameters::numOutput);

        initializeSparse();


        float epsilon = 0.003f;
        {
            Timer a("Simulation wall clock:");

            // Loop through trials
            for(unsigned int trial = 0; trial < Parameters::numTrials; trial++) {
                if((trial % 100) == 0) {
                    // if this isn't the first trial, reduce learning rate
                    if(trial != 0) {
                        learningRate *= 0.7f;
                    }

                    std::cout << "Trial " << trial << " (learning rate " << learningRate << ")" << std::endl;
                }

                // Loop through timesteps within trial
                for(unsigned int i = 0; i < trialTimesteps; i++) {
                    stepTime();

                    // If it's time to update weights
                    if((iT % updateTimesteps) == 0) {
                        BatchLearning::rMaxPropCUDA(d_mInput_Hidden, d_upsilonInput_Hidden, d_wInput_Hidden,
                                                 Parameters::numInput, Parameters::numHidden,
                                                 Parameters::updateTimeMs, Parameters::tauRMS, Parameters::r0, epsilon, Parameters::wMin, Parameters::wMax);
                        BatchLearning::rMaxPropTransposeCUDA(d_mHidden_Output, d_upsilonHidden_Output, d_wHidden_Output,
                                                             d_wOutput_Hidden, Parameters::numHidden, Parameters::numOutput,
                                                             Parameters::updateTimeMs, Parameters::tauRMS, Parameters::r0, epsilon, Parameters::wMin, Parameters::wMax);
                    }

                }

        }
        // Open CSV output files
        /*SpikeRecorder<SpikeWriterTextCached> spikes(&getECurrentSpikes, &getECurrentSpikeCount, "spikes.csv", ",", true);

        {
            Timer a("Simulation wall clock:");
            while(t < 10000.0) {
                // Simulate
                stepTime();

                pullECurrentSpikesFromDevice();


                spikes.record(t);
            }
        }

        spikes.writeCache();*/

        std::cout << "Init:" << initTime << std::endl;
        std::cout << "Init sparse:" << initSparseTime << std::endl;
        std::cout << "Neuron update:" << neuronUpdateTime << std::endl;
        std::cout << "Presynaptic update:" << presynapticUpdateTime << std::endl;
        std::cout << "Synapse dynamics:" << synapseDynamicsTime << std::endl;

        return EXIT_SUCCESS;
    }
}
