// Standard C++ includes
#include <algorithm>
#include <bitset>
#include <iostream>
#include <numeric>
#include <random>

// GeNN user project includes
#include "timer.h"
#include "spikeRecorder.h"

// GeNN generated code includes
#include "izhikevich_pavlovian_CODE/definitions.h"

// Model includes
#include "parameters.h"

//------------------------------------------------------------------------
// Anonymous namespace
//------------------------------------------------------------------------
namespace
{
unsigned int convertMsToTimesteps(double ms)
{
    return (unsigned int)std::round(ms / Parameters::timestepMs);
}

template<unsigned int NumPre>
std::pair<float, float> getMeanOutgoingWeight(unsigned int maxRowLength, const unsigned int *rowLength, const scalar *weights,
                                              const std::bitset<NumPre> &rewardedNeuronSet) {
    // Loop through pre-synaptic neurons
    scalar totalOutgoingWeight = 0.0f;
    scalar totalRewardedOutgoingWeight = 0.0f;
    for(unsigned int i = 0; i < NumPre; i++) {
        const size_t rowStart = (i * maxRowLength);
        const size_t rowEnd = rowStart + rowLength[i];
        
        // Sum weights in row
        const scalar totalRowOutgoingWeight = std::accumulate(&weights[rowStart], &weights[rowEnd], 0.0f);

        // Divide by row length to get mean outgoing weight from this presynaptic neuron
        const scalar meanRowOutgoingWeight = totalRowOutgoingWeight / (float)(rowEnd - rowStart);

        // Add to total
        totalOutgoingWeight += meanRowOutgoingWeight;

        // If this presynaptic neuron is in the rewarded neuron set, add the mean to this total as well
        if(rewardedNeuronSet.test(i)) {
            totalRewardedOutgoingWeight += meanRowOutgoingWeight;
        }
     }

     // Divide total by number of presynaptic neurons and total rewarded
     // by number of rewarded neurons to get averages and return
     return std::make_pair(totalOutgoingWeight / (float)NumPre,
                           totalRewardedOutgoingWeight / (float)rewardedNeuronSet.count());
}
}

int main()
{
    std::mt19937 gen;

    allocateMem();
    initialize();
    initializeSparse();

    std::vector<std::vector<unsigned int>> inputSets;
    std::bitset<Parameters::numExcitatory> rewardedExcStimuliSet;

    {
        Timer timer("Stimuli generation:");

        // Resize input sets vector
        inputSets.resize(Parameters::numStimuliSets);

        // Build array of neuron indices
        std::vector<unsigned int> neuronIndices(Parameters::numCells);
        std::iota(neuronIndices.begin(), neuronIndices.end(), 0);

        // Loop through input sets
        for(auto &i : inputSets) {
            // Shuffle neuron indices
            std::shuffle(neuronIndices.begin(), neuronIndices.end(), gen);

            // Copy first set size indices into input set
            i.resize(Parameters::stimuliSetSize);
            std::copy_n(neuronIndices.begin(), Parameters::stimuliSetSize, i.begin());
        }

        // Create bitset version of excitatory neurons in rewarded stimuli set
        for(unsigned int n : inputSets[0]) {
            if(n < Parameters::numExcitatory) {
                rewardedExcStimuliSet.set(n);
            }
        }
    }

    // Open CSV output files
    SpikeRecorderCached e_spikes("e_spikes.csv", glbSpkCntE, glbSpkE, ",", true);
    SpikeRecorderCached i_spikes("i_spikes.csv", glbSpkCntI, glbSpkI, ",", true);

    std::ofstream stimulusStream("stimulus_times.csv");
    std::ofstream rewardStream("reward_times.csv");
    std::ofstream weightEvolutionStream("weight_evolution.csv");

    {
        Timer timer("Simulation:");

        // Create distribution to pick inter stimuli intervals
        std::uniform_int_distribution<> interStimuliInterval(convertMsToTimesteps(Parameters::minInterStimuliIntervalMs),
                                                             convertMsToTimesteps(Parameters::maxInterStimuliIntervalMs));

        std::uniform_int_distribution<> stimuliSet(0, Parameters::numStimuliSets - 1);

        std::uniform_int_distribution<> rewardDelay(0, (unsigned int)std::round(Parameters::rewardDelayMs / Parameters::timestepMs));

        // Draw time until first stimuli and which set that should be
        unsigned int nextStimuliTimestep = interStimuliInterval(gen);
        unsigned int nextStimuliSet = stimuliSet(gen);

        // Invalidate next reward timestep
        unsigned int nextRewardTimestep = std::numeric_limits<unsigned int>::max();

        // Convert simulation regime parameters to timesteps
        const unsigned int duration = convertMsToTimesteps(Parameters::durationMs);
        const unsigned int recordBeginningStop = convertMsToTimesteps(Parameters::recordStartMs);
        const unsigned int recordEndStart = convertMsToTimesteps(Parameters::durationMs - Parameters::recordEndMs);
        const unsigned int weightRecordInterval = convertMsToTimesteps(Parameters::weightRecordIntervalMs);

        // Loop through timesteps
        while(iT < duration) {
            // Are we in one of the stages of the simulation where we should record spikes
            const bool shouldRecordSpikes = (iT < recordBeginningStop) || (iT > recordEndStart);
            const bool shouldStimulate = (iT == nextStimuliTimestep);
            const bool shouldReward = (iT == nextRewardTimestep);

            // If we should be applying stimuli this timestep
            if(shouldStimulate) {
                std::cout << "\tApplying stimuli set " << nextStimuliSet << " at time " << t << std::endl;

                // Zero
                std::fill_n(IextE, Parameters::numExcitatory, 0.0f);
                std::fill_n(IextI, Parameters::numInhibitory, 0.0f);

                // Loop through neurons in input set and add stimuli current
                for(unsigned int n : inputSets[nextStimuliSet]) {
                    if(n < Parameters::numExcitatory) {
                        IextE[n] += (scalar)Parameters::stimuliCurrent;
                    }
                    else {
                        IextI[n] += (scalar)Parameters::stimuliCurrent;
                    }
                }

                // Upload stimuli input to GPU
                pushIextEToDevice();
                pushIextIToDevice();

                // Record stimulus time and set
                if(shouldRecordSpikes) {
                    stimulusStream << t << "," << nextStimuliSet << std::endl;
                }

                // If this is the rewarded stimuli
                if(nextStimuliSet == 0) {
                    // Draw time until next reward
                    nextRewardTimestep = iT + rewardDelay(gen);

                    std::cout << "\t\tRewarding at timestep " << nextRewardTimestep << std::endl;
                }

                // Pick time and set for next stimuli
                nextStimuliTimestep = iT + interStimuliInterval(gen);
                nextStimuliSet = stimuliSet(gen);
            }

            // If we should reward in this timestep, inject dopamine
            if(shouldReward) {
                std::cout << "\tApplying reward at time " << t << std::endl;
                injectDopamineEE = true;
                injectDopamineEI = true;

                // Record reward time
                if(shouldRecordSpikes) {
                    rewardStream << t << std::endl;
                }
            }

            // Simulate on GPU
            stepTime();

            // If we should be recording spikes, download them from GPU
            if(shouldRecordSpikes) {
                pullECurrentSpikesFromDevice();
                pullICurrentSpikesFromDevice();
            }

            // If we should record weights this time step, download them from GPU
           /* if((t % weightRecordInterval) == 0) {
                CHECK_CUDA_ERRORS(cudaMemcpy(gEE, d_gEE, CEE.connN * sizeof(scalar), cudaMemcpyDeviceToHost));
                CHECK_CUDA_ERRORS(cudaMemcpy(gEI, d_gEI, CEI.connN * sizeof(scalar), cudaMemcpyDeviceToHost));
            }*/

            // If a dopamine spike has been injected this timestep
            if(shouldReward) {
                // Decay global dopamine traces
                dEE = dEE * std::exp(-t / Parameters::tauD);
                dEI = dEI * std::exp(-t / Parameters::tauD);

                // Add effect of dopamine spike
                dEE += Parameters::dopamineStrength;
                dEI += Parameters::dopamineStrength;

                // Update last reward time
                tDEE = t;
                tDEI = t;

                // Clear dopamine injection flags
                injectDopamineEE = false;
                injectDopamineEI = false;
            }

            // If stimulation was applied this timestep
            if(shouldStimulate) {
                // Re-zero external stimuli arrays
                std::fill_n(IextE, Parameters::numExcitatory, 0.0f);
                std::fill_n(IextI, Parameters::numInhibitory, 0.0f);

                // Upload stimuli input to GPU
                pushIextEToDevice();
                pushIextIToDevice();
            }
             // If we should record weights this time step
            /*if((t % weightRecordInterval) == 0) {
                // Calculate the mean outgoing weights within the EE and EI projections
                auto eeOutgoing = getMeanOutgoingWeight<Parameters::numExcitatory>(CEE, gEE, rewardedExcStimuliSet);
                auto eiOutgoing = getMeanOutgoingWeight<Parameters::numExcitatory>(CEI, gEI, rewardedExcStimuliSet);

                // Take the average of these two and write to file
                weightEvolutionStream << (eeOutgoing.first + eiOutgoing.first) / 2.0f << ","<< (eeOutgoing.second + eiOutgoing.second) / 2.0f << std::endl;

            }*/

            // If we should be recording spikes, write spikes to file
            if(shouldRecordSpikes) {
                e_spikes.record(t);
                i_spikes.record(t);
            }
        }
    }
    
    // Write spike data to disk
    e_spikes.writeCache();
    i_spikes.writeCache();

    std::cout << "Init:" << initTime << std::endl;
    std::cout << "Init sparse:" << initSparseTime << std::endl;
    std::cout << "Neuron update:" << neuronUpdateTime << std::endl;
    std::cout << "Presynaptic update:" << presynapticUpdateTime << std::endl;
    std::cout << "Postsynaptic update:" << postsynapticUpdateTime << std::endl;

    return 0;
}
