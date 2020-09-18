// Standard C++ includes
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

// GeNN userproject includes
#include "timer.h"
#include "spikeRecorder.h"

// Batch learning includes
#include "batch_learning.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "superspike_demo_CODE/definitions.h"

namespace
{
void stripWindowsLineEnding(std::string &lineString)
{
    // If line has a Windows line ending, remove it
    if(!lineString.empty() && lineString.back() == '\r') {
        lineString.pop_back();
    }
}
void loadSpikes(const std::string &filename)
{
    // Open ras file
    std::ifstream rasFile(filename);
    if(!rasFile.good()) {
        throw std::runtime_error("Cannot open ras file: " + filename);
    }

    // Read lines into strings
    std::vector<std::pair<unsigned int, double>> data;
    std::string lineString;
    while(std::getline(rasFile, lineString)) {
        // Strip windows line endings
        stripWindowsLineEnding(lineString);

        // Wrap line in stream for easier parsing
        std::istringstream lineStream(lineString);

        // Add new tuple to vector and read line into it
        data.emplace_back();
        lineStream >> data.back().second;
        lineStream >> data.back().first;

        std::cout << data.back().first << "," << data.back().second << std::endl;
    }
}
}   // Anonymous namespace
int main()
{
    try
    {
        loadSpikes("oxford-target.ras");
        allocateMem();
        initialize();

        // Use CUDA to calculate initial transpose of feedforward hidden->output weights
        BatchLearning::transposeCUDA(d_wHidden_Output, d_wOutput_Hidden,
                                    Parameters::numHidden, Parameters::numOutput);

        initializeSparse();


        float epsilon = 0.000000000000000000001f;
        {
            Timer a("Simulation wall clock:");

            // Loop through trials
            for(unsigned int trial = 0; trial < Parameters::numTrials; trial++) {
                if((trial % 100) == 0) {
                    // if this isn't the first trial, reduce learning rate
                    /*if(trial != 0) {
                        learningRate *= 0.7f;
                    }*/

                    std::cout << "Trial " << trial << " (epsilon " << epsilon << ")" << std::endl;
                }

                // Loop through timesteps within trial
                for(unsigned int i = 0; i < Parameters::trialTimesteps; i++) {
                    stepTime();

                    // If it's time to update weights
                    if((iT % Parameters::updateTimesteps) == 0) {
                        BatchLearning::rMaxPropCUDA(d_mInput_Hidden, d_upsilonInput_Hidden, d_wInput_Hidden,
                                                 Parameters::numInput, Parameters::numHidden,
                                                 Parameters::updateTimeMs, Parameters::tauRMS, Parameters::r0, epsilon, Parameters::wMin, Parameters::wMax);
                        BatchLearning::rMaxPropTransposeCUDA(d_mHidden_Output, d_upsilonHidden_Output, d_wHidden_Output,
                                                             d_wOutput_Hidden, Parameters::numHidden, Parameters::numOutput,
                                                             Parameters::updateTimeMs, Parameters::tauRMS, Parameters::r0, epsilon, Parameters::wMin, Parameters::wMax);
                    }

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
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
