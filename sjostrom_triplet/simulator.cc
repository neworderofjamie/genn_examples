#include <algorithm>

#include "sjostrom_triplet_CODE/definitions.h"
#include "parameters.h"

int main()
{
    std::cout << "Num neurons:" << Parameters::numNeurons << std::endl;
    allocateMem();

    // 1-1 connecting stimuli to neurons
    allocatePreStimToExcitatory(Parameters::numNeurons);
    allocatePostStimToExcitatory(Parameters::numNeurons);

    initialize();

    // Loop through connections
    for(unsigned int i = 0; i < Parameters::numNeurons; i++)
    {
        // Each presynaptic neuron only has
        // one postsynaptic neuron connected to it
        CPreStimToExcitatory.indInG[i] = i;
        CPostStimToExcitatory.indInG[i] = i;

        // And this postsynaptic neuron has the same number
        CPreStimToExcitatory.ind[i] = i;
        CPostStimToExcitatory.ind[i] = i;

        gPreStimToExcitatory[i] = 0.5;
        gPostStimToExcitatory[i] = 2.0;
    }

    CPreStimToExcitatory.indInG[Parameters::numNeurons] = Parameters::numNeurons;
    CPostStimToExcitatory.indInG[Parameters::numNeurons] = Parameters::numNeurons;

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

    FILE *spikes = fopen("spikes.csv", "w");
    fprintf(spikes, "Time(ms), Neuron ID\n");

    // Loop through timesteps
    for(unsigned int t = 0; t < simTimesteps; t++)
    {
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

                // **YUCK** also update the time used for post-after-pre STDP calculations
                sTPreStim[n] = 1.0 * (double)t;

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
        stepTimeCPU();

        // Write spike times to file
        for(unsigned int i = 0; i < glbSpkCntExcitatory[0]; i++)
        {
            fprintf(spikes, "%f, %u\n", 1.0 * (double)t, glbSpkExcitatory[i]);
        }
    }
    fclose(spikes);

    FILE *weights = fopen("weights.csv", "w");
    fprintf(weights, "Frequency [Hz], Delta T [ms], Weight\n");

    for(unsigned int n = 0; n < Parameters::numNeurons; n++)
    {
        fprintf(weights, "%f, %f, %f\n", Parameters::frequencies[n / 2], Parameters::dt[n % 2], gPreStimToExcitatory[n]);
    }

    fclose(weights);


    return 0;
}