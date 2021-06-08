#include "stdp_curve_CODE/definitions.h"

// Standard C++ includes
#include <iostream>

// GeNN userproject includes
#include "spikeRecorder.h"

#define NUM_NEURONS 14
#define NUM_PAIRS 60

int main()
{
    allocateMem();

    initialize();

    // Initialise pre and postsynaptic stimuli variables - each one should emit 60 spikes
    for(unsigned int n = 0; n < NUM_NEURONS; n++) {
        startSpikePostStim[n] = n * 60;
        startSpikePreStim[n] = n * 60;
        endSpikePostStim[n] = (n + 1) * 60;
        endSpikePreStim[n] = (n + 1) * 60;
    }

    // Setup reverse connection indices for STDP
    initializeSparse();

    // Allocate memory for spike source array
    allocatespikeTimesPostStim(NUM_PAIRS * NUM_NEURONS);
    allocatespikeTimesPreStim(NUM_PAIRS * NUM_NEURONS);

    // Spike pair configuration
    const scalar startTime = 200.0;
    const scalar timeBetweenPairs = 1000.0;
    const scalar deltaT[NUM_NEURONS] = {-100.0, -60.0, -40.0, -30.0, -20.0, -10.0, -1.0,
                                        1.0, 10.0, 20.0, 30.0, 40.0, 60.0, 100.0};

    // Loop through neurons
    for(unsigned int n = 0; n < NUM_NEURONS; n++)
    {
        // Calculate spike timings
        const double neuronDeltaT = deltaT[n];
        const double prePhase = (neuronDeltaT > 0) ? (startTime + neuronDeltaT + 1.0) : (startTime + 1.0);
        const double postPhase = (neuronDeltaT > 0) ? startTime : (startTime - neuronDeltaT);

        std::cout << "Neuron " << n << "(" << neuronDeltaT << ")): pre phase " << prePhase << " , post phase " << postPhase << std::endl;

        // Fill in spike timings
        scalar *postStimSpikes = &spikeTimesPostStim[n * 60];
        scalar *preStimSpikes = &spikeTimesPreStim[n * 60];
        for(unsigned int p = 0; p < 60; p++) {
            (*postStimSpikes++) = postPhase + ((scalar)p * timeBetweenPairs);
            (*preStimSpikes++) = prePhase + ((scalar)p * timeBetweenPairs);
        }
    }

    // Upload spike times
    pushspikeTimesPostStimToDevice(NUM_PAIRS * NUM_NEURONS);
    pushspikeTimesPreStimToDevice(NUM_PAIRS * NUM_NEURONS);

    // Loop through timesteps
    SpikeRecorder<> spikes(&getExcitatoryCurrentSpikes, &getExcitatoryCurrentSpikeCount, 
                           "spikes.csv", ", ", true);
    while(iT < 60200) {
        // Simulate
        stepTime();

        pullExcitatoryCurrentSpikesFromDevice();
        spikes.record(t);
    }

    // Download weights
    pullgPreStimToExcitatoryFromDevice();

    // Write weights to CSV
    std::ofstream weights("weights.csv");
    weights << "Delta T [ms], Weight" << std::endl;
    for(unsigned int n = 0; n < NUM_NEURONS; n++) {
        weights << deltaT[n] << ", " << gPreStimToExcitatory[n] << std::endl;
    }

    // Free spike times
    freespikeTimesPostStim();
    freespikeTimesPreStim();

    return 0;
}
