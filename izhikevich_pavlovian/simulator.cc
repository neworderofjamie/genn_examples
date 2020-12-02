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
    // Convert simulation regime parameters to timesteps
    const unsigned int duration = convertMsToTimesteps(Parameters::durationMs);
    const unsigned int recordTime = convertMsToTimesteps(Parameters::recordTimeMs);
    const unsigned int weightRecordInterval = convertMsToTimesteps(Parameters::weightRecordIntervalMs);

    // Assert that duration is a multiple of record time
    assert((duration % recordTime) == 0);

    std::mt19937 gen;

    allocateMem();
    allocateRecordingBuffers(recordTime);
    initialize();

    // Allocate a bit per timestep for dopamine injection times
    const unsigned int numTimestepWords = (duration + 31) / 32;
    allocatedTimeE(numTimestepWords);
    
    // Initially zero all dopamine injection times
    std::fill_n(dTimeE, numTimestepWords, 0);
    
    std::bitset<Parameters::numExcitatory> rewardedExcStimuliSet;
    {
        Timer timer("Stimuli generation:");

        // Resize input sets vector
        std::vector<std::vector<unsigned int>> inputSets;
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
        
        // Create distributions to pick inter stimuli intervals, stimuli sets and reward delays
        std::uniform_int_distribution<> interStimuliIntervalDist(convertMsToTimesteps(Parameters::minInterStimuliIntervalMs),
                                                                 convertMsToTimesteps(Parameters::maxInterStimuliIntervalMs));

        std::uniform_int_distribution<> stimuliSetDist(0, Parameters::numStimuliSets - 1);

        std::uniform_int_distribution<> rewardDelayDist(0, (unsigned int)std::round(Parameters::rewardDelayMs / Parameters::timestepMs));

        // Draw time until first stimuli and which set that should be
        unsigned int nextStimuliTimestep = interStimuliIntervalDist(gen);

        // Allocate vector of vectors to hold stimuli times
        std::vector<std::vector<float>> neuronStimuliTimes;
        neuronStimuliTimes.resize(Parameters::numCells);
        
        std::ofstream stimulusStream("stimulus_times.csv");
        std::ofstream rewardStream("reward_times.csv");
    
        // While we're within duration
        size_t totalNumEStimuli = 0;
        size_t totalNumIStimuli = 0;
        while(nextStimuliTimestep < duration) {
            // Determine what stimuli to present
            const unsigned int stimululiSet = stimuliSetDist(gen);
            
            // Loop through neurons to stimulate and add spike times to correct vector
            for(unsigned int n : inputSets[stimululiSet]) {
                neuronStimuliTimes[n].push_back((float)nextStimuliTimestep * (float)Parameters::timestepMs);
                
                if(n < Parameters::numExcitatory) {
                    totalNumEStimuli++;
                }
                else {
                    totalNumIStimuli++;
                }
            }
            
            // If we should be recording at this point, write stimuli to file
            if((nextStimuliTimestep < recordTime) || (nextStimuliTimestep > (duration - recordTime))) {
                stimulusStream << nextStimuliTimestep << "," << stimululiSet << std::endl;
            }
            
            // If this is the rewarded stimuli
            if(stimululiSet == 0) {
                // Draw time until next reward
                const unsigned int rewardTimestep = nextStimuliTimestep + rewardDelayDist(gen);
                
                // Set bit in dopamine times bitset
                if(rewardTimestep < duration) {
                    dTimeE[rewardTimestep / 32] |= (1 << (rewardTimestep % 32));
                    
                    // If we should be recording at this point, write reward to file
                    if((rewardTimestep < recordTime) || (rewardTimestep > (duration - recordTime))) {
                        rewardStream << rewardTimestep << std::endl;
                    }
                }
            }

            // Advance to next stimuli
            nextStimuliTimestep += interStimuliIntervalDist(gen);
        }
        
        // Upload dopamine data
        pushdTimeEToDevice(numTimestepWords);
        
        {
            // Allocate stimuli times array
            allocatestimTimesECurr(totalNumEStimuli);
            
            unsigned int numStim = 0;
            for(unsigned int i = 0; i < Parameters::numExcitatory; i++) {
                startStimECurr[i] = numStim;
                
                const auto &stimTimes = neuronStimuliTimes[i];
                std::copy(stimTimes.cbegin(), stimTimes.cend(), &stimTimesECurr[numStim]);
                

                numStim += stimTimes.size();
                endStimECurr[i] = numStim;
            }
            assert(numStim == totalNumEStimuli);
            
            // Upload stimuli times to GPU
            pushstimTimesECurrToDevice(totalNumEStimuli);
        }
        {
            
            // Allocate stimuli times array
            allocatestimTimesICurr(totalNumIStimuli);
            
            unsigned int numStim = 0;
            for(unsigned int i = 0; i < Parameters::numInhibitory; i++) {
                startStimICurr[i] = numStim;
                
                const auto &stimTimes = neuronStimuliTimes[i + Parameters::numExcitatory];
                std::copy(stimTimes.cbegin(), stimTimes.cend(), &stimTimesICurr[numStim]);
                

                numStim += stimTimes.size();
                endStimICurr[i] = numStim;
            }
            assert(numStim == totalNumIStimuli);
            
            // Upload stimuli times to GPU
            pushstimTimesICurrToDevice(totalNumIStimuli);
        }
    }

    // Complete initialization
    initializeSparse();
    
    std::ofstream weightEvolutionStream("weight_evolution.csv");

    {
        Timer timer("Simulation:");

        // Loop through timesteps
        while(iT < duration) {
            // Simulate
            stepTime();

            // If we should record weights this time step, download them from GPU
           /* if((t % weightRecordInterval) == 0) {
                CHECK_CUDA_ERRORS(cudaMemcpy(gEE, d_gEE, CEE.connN * sizeof(scalar), cudaMemcpyDeviceToHost));
                CHECK_CUDA_ERRORS(cudaMemcpy(gEI, d_gEI, CEI.connN * sizeof(scalar), cudaMemcpyDeviceToHost));
            }
           
             // If we should record weights this time step
            if((t % weightRecordInterval) == 0) {
                // Calculate the mean outgoing weights within the EE and EI projections
                auto eeOutgoing = getMeanOutgoingWeight<Parameters::numExcitatory>(CEE, gEE, rewardedExcStimuliSet);
                auto eiOutgoing = getMeanOutgoingWeight<Parameters::numExcitatory>(CEI, gEI, rewardedExcStimuliSet);

                // Take the average of these two and write to file
                weightEvolutionStream << (eeOutgoing.first + eiOutgoing.first) / 2.0f << ","<< (eeOutgoing.second + eiOutgoing.second) / 2.0f << std::endl;

            }*/

            // If we've just filled the recording buffer with data we want
            if(iT == recordTime || iT == duration) {
                const bool firstRecordingBlock = (iT == recordTime);
                const double recordingBlockStart = t - Parameters::recordTimeMs;
                
                // Download recording data
                pullRecordingBuffersFromDevice();
                
                // Write spike data to CSV, starting a new file with a header if this is the 
                // first recording phase and appending without a header if it is the second
                writeTextSpikeRecording("e_spikes.csv", recordSpkE, Parameters::numExcitatory, 
                                        recordTime, Parameters::timestepMs, ",", firstRecordingBlock, 
                                        !firstRecordingBlock, recordingBlockStart);
                writeTextSpikeRecording("i_spikes.csv", recordSpkI, Parameters::numInhibitory, 
                                        recordTime, Parameters::timestepMs, ",", firstRecordingBlock,
                                        !firstRecordingBlock, recordingBlockStart);
            }
        }
    }

    if(Parameters::measureTiming) {
        std::cout << "Init:" << initTime << std::endl;
        std::cout << "Init sparse:" << initSparseTime << std::endl;
        std::cout << "Neuron update:" << neuronUpdateTime << std::endl;
        std::cout << "Presynaptic update:" << presynapticUpdateTime << std::endl;
        std::cout << "Postsynaptic update:" << postsynapticUpdateTime << std::endl;
    }

    return 0;
}
