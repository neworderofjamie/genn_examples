// Standard C++ includes
#include <algorithm>
#include <bitset>
#include <numeric>
#include <random>

// Common includes
#include "connectors.h"
#include "../common/spike_csv_recorder.h"
#include "../common/timer.h"

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
std::pair<float, float> getMeanOutgoingWeight(const SparseProjection &projection, const scalar *weights,
                                              const std::bitset<NumPre> &rewardedNeuronSet) {
    // Loop through pre-synaptic neurons
    scalar totalOutgoingWeight = 0.0f;
    scalar totalRewardedOutgoingWeight = 0.0f;
    for(unsigned int i = 0; i < NumPre; i++) {
        // Get indices of row start and end and hence row-length
        const unsigned int rowStart = projection.indInG[i];
        const unsigned int rowEnd = projection.indInG[i + 1];

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

    // Final setup
    {
        Timer<> t("Sparse init:");
        initizhikevich_pavlovian();
    }

    std::vector<std::vector<unsigned int>> inputSets;
    std::bitset<Parameters::numExcitatory> rewardedExcStimuliSet;

    {
        Timer<> t("Stimuli generation:");

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
    SpikeCSVRecorder e_spikes("e_spikes.csv", glbSpkCntE, glbSpkE);
    SpikeCSVRecorder i_spikes("i_spikes.csv", glbSpkCntI, glbSpkI);

    std::ofstream stimulusStream("stimulus_times.csv");
    std::ofstream rewardStream("reward_times.csv");
    std::ofstream weightEvolutionStream("weight_evolution.csv");

    {
        Timer<> t("Simulation:");

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
        for(unsigned int t = 0; t < duration; t++)
        {
            // Are we in one of the stages of the simulation where we should record spikes
            const bool shouldRecordSpikes = (t < recordBeginningStop) || (t > recordEndStart);
            const bool shouldStimulate = (t == nextStimuliTimestep);
            const bool shouldReward = (t == nextRewardTimestep);

            // If we should be applying stimuli this timestep
            if(shouldStimulate) {
                std::cout << "\tApplying stimuli set " << nextStimuliSet << " at timestep " << t << std::endl;

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

#ifndef CPU_ONLY
                // Upload stimuli input to GPU
                CHECK_CUDA_ERRORS(cudaMemcpy(d_IextE, IextE, Parameters::numExcitatory * sizeof(scalar), cudaMemcpyHostToDevice));
                CHECK_CUDA_ERRORS(cudaMemcpy(d_IextI, IextI, Parameters::numInhibitory * sizeof(scalar), cudaMemcpyHostToDevice));
#endif

                // Record stimulus time and set
                if(shouldRecordSpikes) {
                    stimulusStream << (scalar)t * DT << "," << nextStimuliSet << std::endl;
                }

                // If this is the rewarded stimuli
                if(nextStimuliSet == 0) {
                    // Draw time until next reward
                    nextRewardTimestep = t + rewardDelay(gen);

                    std::cout << "\t\tRewarding at timestep " << nextRewardTimestep << std::endl;
                }

                // Pick time and set for next stimuli
                nextStimuliTimestep = t + interStimuliInterval(gen);
                nextStimuliSet = stimuliSet(gen);
            }

            // If we should reward in this timestep, inject dopamine
            if(shouldReward) {
                std::cout << "\tApplying reward at timestep " << t << std::endl;
                injectDopamineEE = true;
                injectDopamineEI = true;

                // Record reward time
                if(shouldRecordSpikes) {
                    rewardStream << (scalar)t * DT << std::endl;
                }
            }

#ifndef CPU_ONLY
            // Simulate on GPU
            stepTimeGPU();

            // If we should be recording spikes, download them from GPU
            if(shouldRecordSpikes) {
                pullECurrentSpikesFromDevice();
                pullICurrentSpikesFromDevice();
            }

            // If we should record weights this time step, download them from GPU
            if((t % weightRecordInterval) == 0) {
                CHECK_CUDA_ERRORS(cudaMemcpy(gEE, d_gEE, CEE.connN * sizeof(scalar), cudaMemcpyDeviceToHost));
                CHECK_CUDA_ERRORS(cudaMemcpy(gEI, d_gEI, CEI.connN * sizeof(scalar), cudaMemcpyDeviceToHost));
            }
#else
            // Simulate on CPU
            stepTimeCPU();
#endif
            // If a dopamine spike has been injected this timestep
            if(shouldReward) {
                const scalar tMs =  (scalar)t * DT;

                // Decay global dopamine traces
                dEE = dEE * std::exp(-tMs / Parameters::tauD);
                dEI = dEI * std::exp(-tMs / Parameters::tauD);

                // Add effect of dopamine spike
                dEE += Parameters::dopamineStrength;
                dEI += Parameters::dopamineStrength;

                // Update last reward time
                tDEE = tMs;
                tDEI = tMs;

                // Clear dopamine injection flags
                injectDopamineEE = false;
                injectDopamineEI = false;
            }
#ifndef CPU_ONLY
            // If stimulation was applied this timestep
            if(shouldStimulate) {
                // Re-zero external stimuli arrays
                std::fill_n(IextE, Parameters::numExcitatory, 0.0f);
                std::fill_n(IextI, Parameters::numInhibitory, 0.0f);

                // Copy them back to GPU
                CHECK_CUDA_ERRORS(cudaMemcpy(d_IextE, IextE, Parameters::numExcitatory * sizeof(scalar), cudaMemcpyHostToDevice));
                CHECK_CUDA_ERRORS(cudaMemcpy(d_IextI, IextI, Parameters::numInhibitory * sizeof(scalar), cudaMemcpyHostToDevice));
            }
#endif
             // If we should record weights this time step
            if((t % weightRecordInterval) == 0) {
                // Calculate the mean outgoing weights within the EE and EI projections
                auto eeOutgoing = getMeanOutgoingWeight<Parameters::numExcitatory>(CEE, gEE, rewardedExcStimuliSet);
                auto eiOutgoing = getMeanOutgoingWeight<Parameters::numExcitatory>(CEI, gEI, rewardedExcStimuliSet);

                // Take the average of these two and write to file
                weightEvolutionStream << (eeOutgoing.first + eiOutgoing.first) / 2.0f << ","<< (eeOutgoing.second + eiOutgoing.second) / 2.0f << std::endl;

            }

            // If we should be recording spikes, write spikes to file
            if(shouldRecordSpikes) {
                e_spikes.record(t);
                i_spikes.record(t);
            }
        }
    }

    return 0;
}
