#include <algorithm>
#include <numeric>
#include <random>

#include "../common/connectors.h"
#include "../common/spike_csv_recorder.h"
#include "../common/timer.h"

#include "izhikevich_pavlovian_CODE/definitions.h"

#include "parameters.h"

int main()
{
    std::mt19937 gen;

    {
        Timer<> t("Allocation:");
        allocateMem();
    }

    {
        Timer<> t("Initialization:");
        initialize();
    }

    {
        Timer<> t("Building connectivity:");
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numInhibitory,
                                    Parameters::probabilityConnection, CII, &allocateII, gen);
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numExcitatory,
                                    Parameters::probabilityConnection, CIE, &allocateIE, gen);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numExcitatory,
                                    Parameters::probabilityConnection, CEE, &allocateEE, gen);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numInhibitory,
                                    Parameters::probabilityConnection, CEI, &allocateEI, gen);
    }

    {
        Timer<> t("Initializing sparse synapse variables:");

        // Initialize excitatory weights
        std::fill_n(gEI, CEI.connN, 1.0f);
        std::fill_n(gEE, CEE.connN, 1.0f);

        // Initialize synaptic tags
        std::fill_n(cEI, CEI.connN, 0.0f);
        std::fill_n(cEE, CEE.connN, 0.0f);

        // Initialize times to last update
        std::fill_n(tCEI, CEI.connN, 0.0f);
        std::fill_n(tCEE, CEE.connN, 0.0f);
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initizhikevich_pavlovian();
    }

    std::vector<std::vector<unsigned int>> inputSets;
    {
        Timer<> t("Stimuli generation:");

        // Resize input sets vector
        inputSets.resize(Parameters::numStimuliSets);

        // Build array of neuron indices
        std::vector<unsigned int> neuronIndices(Parameters::numExcitatory + Parameters::numInhibitory);
        std::iota(neuronIndices.begin(), neuronIndices.end(), 0);

        // Loop through input sets
        for(auto &i : inputSets) {
            // Shuffle neuron indices
            std::shuffle(neuronIndices.begin(), neuronIndices.end(), gen);

            // Copy first set size indices into input set
            i.resize(Parameters::stimuliSetSize);
            std::copy_n(neuronIndices.begin(), Parameters::stimuliSetSize, i.begin());
        }
    }

    // Open CSV output files
    SpikeCSVRecorder e_spikes("e_spikes.csv", glbSpkCntE, glbSpkE);
    SpikeCSVRecorder i_spikes("i_spikes.csv", glbSpkCntI, glbSpkI);

    std::ofstream stimulusStream("stimulus_times.csv");
    std::ofstream rewardStream("reward_times.csv");

    {
        Timer<> t("Simulation:");

        // Create distribution to pick an input to apply thamalic input to
        std::uniform_real_distribution<> inputCurrent(-6.5, 6.5);

        // Create distribution to pick inter stimuli intervals
        std::uniform_int_distribution<> interStimuliInterval((unsigned int)std::round(Parameters::minInterStimuliIntervalMs / Parameters::timestepMs),
                                                            (unsigned int)std::round(Parameters::maxInterStimuliIntervalMs / Parameters::timestepMs));

        std::uniform_int_distribution<> stimuliSet(0, Parameters::numStimuliSets - 1);

        std::uniform_int_distribution<> rewardDelay(0, (unsigned int)std::round(Parameters::rewardDelayMs / Parameters::timestepMs));

        // Draw time until first stimuli and which set that should be
        unsigned int nextStimuliTimestep = interStimuliInterval(gen);
        unsigned int nextStimuliSet = stimuliSet(gen);

        // Invalidate next reward timestep
        unsigned int nextRewardTimestep = std::numeric_limits<unsigned int>::max();

        // Loop through timesteps
        const unsigned int duration = (unsigned int)std::round(Parameters::durationMs / Parameters::timestepMs);
        const unsigned int recordBeginningStop = (unsigned int)std::round(Parameters::recordStartMs / Parameters::timestepMs);
        const unsigned int recordEndStart = (unsigned int)std::round((Parameters::durationMs - Parameters::recordEndMs) / Parameters::timestepMs);
        for(unsigned int t = 0; t < duration; t++)
        {
            const bool shouldRecord = (t < recordBeginningStop) || (t > recordEndStart);

            // Generate uniformly distributed numbers to fill host array
            // **TODO** move to GPU
            std::generate_n(IextE, Parameters::numExcitatory,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });
            std::generate_n(IextI, Parameters::numInhibitory,
                [&inputCurrent, &gen](){ return inputCurrent(gen); });

            // If we should be applying input in
            if(t == nextStimuliTimestep) {
                std::cout << "Applying stimuli set " << nextStimuliSet << " at timestep " << t << std::endl;

                // Loop through neurons in input set and add stimuli current
                for(unsigned int n : inputSets[nextStimuliSet]) {
                    if(n < Parameters::numExcitatory) {
                        IextE[n] += (scalar)Parameters::stimuliCurrent;
                    }
                    else {
                        IextI[n] += (scalar)Parameters::stimuliCurrent;
                    }
                }

                // Record stimulus time and set
                if(shouldRecord) {
                    stimulusStream << (scalar)t * DT << "," << nextStimuliSet << std::endl;
                }

                // If this is the rewarded stimuli
                if(nextStimuliSet == 0) {
                    // Draw time until next reward
                    nextRewardTimestep = t + rewardDelay(gen);

                    std::cout << "\tRewarding at timestep " << nextRewardTimestep << std::endl;
                }

                // Pick time and set for next stimuli
                nextStimuliTimestep = t + interStimuliInterval(gen);
                nextStimuliSet = stimuliSet(gen);
            }

            // If we should reward in this timestep, inject dopamine
            if(t == nextRewardTimestep) {
                std::cout << "Applying reward at timestep " << t << std::endl;
                injectDopamineEE = true;
                injectDopamineEI = true;

                // Record reward time
                if(shouldRecord) {
                    rewardStream << (scalar)t * DT << std::endl;
                }
            }

            // Simulate
#ifndef CPU_ONLY
            // Upload random input currents to GPU
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IextE, IextE, Parameters::numExcitatory * sizeof(scalar), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERRORS(cudaMemcpy(d_IextI, IextI, Parameters::numInhibitory * sizeof(scalar), cudaMemcpyHostToDevice));

            stepTimeGPU();

            // Download spikes from GPU
            if(shouldRecord) {
                pullECurrentSpikesFromDevice();
                pullICurrentSpikesFromDevice();
            }
#else
            stepTimeCPU();
#endif
            if(t == nextRewardTimestep) {
                const scalar tMs =  (scalar)t * DT;

                // Decay global dopamine traces
                dEE = dEE * std::exp(-tMs / Parameters::tauD);
                dEI = dEI * std::exp(-tMs / Parameters::tauD);

                // Add effect of dopamine spike
                dEE += 0.5f;
                dEI += 0.5f;

                // Update last reward time
                tDEE = tMs;
                tDEI = tMs;

                // Clear dopamine injection flags
                injectDopamineEE = false;
                injectDopamineEI = false;
            }

            // Record spikes
            if(shouldRecord) {
                e_spikes.record(t);
                i_spikes.record(t);
            }
        }
    }

    return 0;
}
