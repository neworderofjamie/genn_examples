#include <algorithm>

#include "sjostrom_triplet_CODE/definitions.h"

#include "genn_utils/spike_csv_recorder.h"

#include "parameters.h"

using namespace BoBRobotics;

int main()
{
    std::cout << "Num neurons:" << Parameters::numNeurons << std::endl;
    allocateMem();

    initialize();

    // Setup reverse connection indices for STDP
    initsjostrom_triplet();

    // Spike pair configuration
    const double startTime = 100.0;
    const unsigned int numTriplets = 60;

    const unsigned int numPreSpikes = numTriplets + 1;
    const unsigned int numPostSpikes = numTriplets;

    // Loop through neurons
    unsigned int preSpikeTimesteps[Parameters::numNeurons][numPreSpikes];
    unsigned int postSpikeTimesteps[Parameters::numNeurons][numPostSpikes];
    unsigned int nextPreSpikeIndex[Parameters::numNeurons];
    unsigned int nextPostSpikeIndex[Parameters::numNeurons];
    unsigned int simTimesteps = 0;
    for(unsigned int n = 0; n < Parameters::numNeurons; n++)
    {
        // Start each spike source at first spike
        nextPreSpikeIndex[n] = 0;
        nextPostSpikeIndex[n] = 0;

        // Calculate spike timings
        const unsigned int interspikeDelay = std::ceil(1000.0 / Parameters::frequencies[n / 2]);

        // Fill in spike timings
        const unsigned int prePhase = startTime - 1;
        for(unsigned int p = 0; p < numPreSpikes; p++)
        {
            preSpikeTimesteps[n][p] = prePhase + (p * interspikeDelay);
        }

        const unsigned int postPhase = startTime + Parameters::dt[n % 2];
        for(unsigned int p = 0; p < numPostSpikes; p++) {
            postSpikeTimesteps[n][p] = postPhase + (p * interspikeDelay);
        }

        simTimesteps = std::max(simTimesteps, *std::max_element(&preSpikeTimesteps[n][0], &preSpikeTimesteps[n][61]));
        simTimesteps = std::max(simTimesteps, *std::max_element(&postSpikeTimesteps[n][0], &postSpikeTimesteps[n][60]));
    }

    std::cout << "Sim timesteps:" << simTimesteps << std::endl;

    GeNNUtils::SpikeCSVRecorder recorder("spikes.csv", glbSpkCntPre, glbSpkPre);

    // Loop through timesteps
    while(iT < simTimesteps)
    {
        // Zero spike counts
        glbSpkCntPreStim[0] = 0;
        glbSpkCntPostStim[0] = 0;

        // Loop through spike sources
        for(unsigned int n = 0; n < Parameters::numNeurons; n++)
        {
            // If there are more pre-spikes to emit and
            // the next one should be emitted this timestep
            if(nextPreSpikeIndex[n] < numPreSpikes
                && preSpikeTimesteps[n][nextPreSpikeIndex[n]] == t)
            {
                // Manually add a spike to spike source's output
                glbSpkPreStim[glbSpkCntPreStim[0]++] = n;

                // Go onto next pre-spike
                nextPreSpikeIndex[n]++;
            }

            // If there are more post-spikes to emit and
            // the next one should be emitted this timestep
            if(nextPostSpikeIndex[n] < numPostSpikes
                && postSpikeTimesteps[n][nextPostSpikeIndex[n]] == t)
            {
                // Manually add a spike to spike source's output
                glbSpkPostStim[glbSpkCntPostStim[0]++] = n;

                // Go onto next post-spike
                nextPostSpikeIndex[n]++;
            }
        }

        // Simulate
#ifndef CPU_ONLY
        pushPreStimCurrentSpikesToDevice();
        pushPostStimCurrentSpikesToDevice();

        stepTimeGPU();

        pullPreCurrentSpikesFromDevice();
#else
        stepTimeCPU();
#endif

        recorder.record(t);
    }

    FILE *weights = fopen("weights.csv", "w");
    fprintf(weights, "Frequency [Hz], Delta T [ms], Weight\n");

#ifndef CPU_ONLY
    pullPreToPostStateFromDevice();
#endif

    for(unsigned int n = 0; n < Parameters::numNeurons; n++)
    {
        fprintf(weights, "%f, %f, %f\n", Parameters::frequencies[n / 2], Parameters::dt[n % 2], gPreToPost[n]);
    }

    fclose(weights);


    return 0;
}