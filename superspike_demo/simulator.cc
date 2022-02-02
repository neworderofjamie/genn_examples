// Standard C++ includes
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <sstream>

// GeNN userproject includes
#include "timer.h"
#include "spikeRecorder.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "superspike_demo_CODE/definitions.h"
#include "superspike_demo_CODE/macroLookup.h"

namespace
{
void loadTargetSpikes(const std::string &filename)
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
        // Wrap line in stream for easier parsing
        std::istringstream lineStream(lineString);

        // Add new pair to vector and read line into it
        data.emplace_back();
        lineStream >> data.back().second;
        lineStream >> data.back().first;

        // Make neuron indices zero-based and convert time to ms
        data.back().first--;
        data.back().second *= 1000.0;
    }

    // Sort data
    // **NOTE** std::pair < operator means this will sort by neuron then time
    std::sort(data.begin(), data.end());

    // Allocate memory for spike times
    ALLOCATE_EGP_FIELD(Output, spikeTimes, data.size());

    // Copy just the sorted spike times into this memory and push to device
    std::transform(data.cbegin(), data.cend(), GET_FIELD(Output, spikeTimes),
                   [](const std::pair<unsigned int, double> &s){ return s.second; });
    PUSH_EGP_FIELD(Output, spikeTimes, data.size());

    // Loop through output neurons
    unsigned int spike = 0;
    for(unsigned int i = 0; i < Parameters::numOutput; i++) {
        // Fast-forward until there's a spike from this neuron
        while(spike < data.size() && data[spike].first < i) {
            spike++;
        }

        // Record neurons starting spike index
        GET_FIELD(Output, startSpike)[i] = spike;
        
        // Fast-forward through all this neuron's spikes
        while(spike < data.size() && data[spike].first == i) {
            spike++;
        }

        // Record neurons ending spike index
        GET_FIELD(Output, endSpike)[i] = spike;
    }

}

void generateFrozenPoissonInput(std::mt19937 &gen)
{
    std::exponential_distribution<float> dist(1.0);

    // Calcualte inter-spike-interval
    const float isiMs = 1000.0f / Parameters::inputFreqHz;

    // Loop through input neurons
    std::vector<float> spikeTimes;
    for(unsigned int i = 0; i < Parameters::numInput; i++) {
        // Record neurons starting spike index
        GET_FIELD(Input, startSpike)[i] = spikeTimes.size();

        // Generate spike train using exponential distribution
        for(float t = isiMs * dist(gen); t < Parameters::trialMs; t += isiMs * dist(gen)) {
            spikeTimes.push_back(t);
        }

        // Record neurons ending spike index
        GET_FIELD(Input, endSpike)[i] = spikeTimes.size();
    }

    // Allocate memory for spike times
    ALLOCATE_EGP_FIELD(Input, spikeTimes, spikeTimes.size());
    std::copy(spikeTimes.cbegin(), spikeTimes.cend(), GET_FIELD(Input, spikeTimes));
    PUSH_EGP_FIELD(Input, spikeTimes, spikeTimes.size());
}

float calculateError(unsigned int timestep)
{
    constexpr double a = Parameters::tauDecay / 1000.0;
    constexpr double b = Parameters::tauRise / 1000.0;
    constexpr double c = Parameters::tauAvgErr / 1000.0;
    const double scaleTrErrFlt = 1.0 / (std::pow((a*b)/(a-b),2)*(a/2+b/2-2*(a*b)/(a+b))) / c;

    const double timeS = timestep * Parameters::timestepMs / 1000.0;

    // Calculate mean error
    scalar *avgSqrErrOutput = GET_FIELD(Output, avgSqrErr);
    const float meanError = std::accumulate(&avgSqrErrOutput[0], &avgSqrErrOutput[Parameters::numOutput], 0.0f) / (float)Parameters::numOutput;
    return scaleTrErrFlt * meanError / (1.0 - std::exp(-timeS / c) + 1.0E-9);
}

void setR0(float r0) 
{
    *GET_FIELD(HiddenOutputWeightOptimiser, r0) = r0;
    *GET_FIELD(InputHiddenWeightOptimiser, r0) = r0;
    PUSH_EGP_FIELD(HiddenOutputWeightOptimiser, r0, 1);
    PUSH_EGP_FIELD(InputHiddenWeightOptimiser, r0, 1);
}
}   // Anonymous namespace

int main()
{
    try
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        allocateMem();
        allocateRecordingBuffers(Parameters::trialTimesteps);
        initialize();

        // Load target spikes
        loadTargetSpikes("oxford-target.ras");
        
        // Generate frozen Poisson input
        generateFrozenPoissonInput(gen);

        // Allocate single elemenr0HiddenOutputWeightOptimisert arrays for r0 EGPs
        ALLOCATE_EGP_FIELD(HiddenOutputWeightOptimiser, r0, 1);
        ALLOCATE_EGP_FIELD(InputHiddenWeightOptimiser, r0, 1);
        
        
        initializeSparse();

        // Calculate initial transpose
        updateCalculateTranspose();
        {
            Timer a("Simulation wall clock:");

            // Loop through trials
            unsigned int timestep = 0;
            
            float r0 = Parameters::r0;
            setR0(r0);
            
            for(unsigned int trial = 0; trial < Parameters::numTrials; trial++) {
                // Reduce learning rate every 400 trials
                if(trial != 0 && (trial % 400) == 0) {
                    r0 *= 0.1;
                    setR0(r0);
                }

                // Display trial number peridically
                if(trial != 0 && (trial % 10) == 0) {
                    // Get average square error
                    PULL_FIELD(Output, avgSqrErr);
                    std::cout << "Trial " << trial << " (r0 = " << r0 << ", error = " << calculateError(timestep) << ")" << std::endl;
                }

                // Reset model timestep
                // **NOTE** this a bit gross but means we can simplify a lot of logic
                t = 0.0f;
                iT = 0;

                // Loop through timesteps within trial
                for(unsigned int i = 0; i < Parameters::trialTimesteps; i++) {
                    stepTime();

                    // If it's time to update weights
                    if(timestep != 0 && (timestep % Parameters::updateTimesteps) == 0) {
                        updateGradientLearn();
                    }

                    timestep++;

                }

                // Reset spike sources by re-uploading starting spike indices
                // **TODO** build repeating spike source array
                PUSH_FIELD(Input, startSpike);
                PUSH_FIELD(Output, startSpike);

                if((trial % 100) == 0) {
                    pullRecordingBuffersFromDevice();
                    writeTextSpikeRecording("input_spikes_" + std::to_string(trial) + ".csv", GET_FIELD(Input, recordSpk),
                                            Parameters::numInput, Parameters::trialTimesteps, Parameters::timestepMs,
                                            ",", true);
                    writeTextSpikeRecording("hidden_spikes_" + std::to_string(trial) + ".csv", GET_FIELD(Hidden, recordSpk),
                                            Parameters::numHidden, Parameters::trialTimesteps, Parameters::timestepMs,
                                            ",", true);
                    writeTextSpikeRecording("output_spikes_" + std::to_string(trial) + ".csv", GET_FIELD(Output, recordSpk),
                                            Parameters::numOutput, Parameters::trialTimesteps, Parameters::timestepMs,
                                            ",", true);
                }

            }

        }

        std::cout << "Init:" << initTime << std::endl;
        std::cout << "Init sparse:" << initSparseTime << std::endl;
        std::cout << "Neuron update:" << neuronUpdateTime << std::endl;
        std::cout << "Presynaptic update:" << presynapticUpdateTime << std::endl;
        std::cout << "Synapse dynamics:" << synapseDynamicsTime << std::endl;
        std::cout << "Gradient learning custom update:" << customUpdateGradientLearnTime << std::endl;
        std::cout << "Gradient learning custom update transpose:" << customUpdateGradientLearnTransposeTime << std::endl;
        return EXIT_SUCCESS;
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
