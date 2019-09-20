#include "if_curr_CODE/definitions.h"

// GeNN userproject includes
#include "analogueRecorder.h"

int main()
{
    allocateMem();
    initialize();
    initializeSparse();

    AnalogueRecorder<float> recorder("voltages.csv", {VExcitatory, inSynStimToExcitatory}, 1, ",");

    unsigned int spikeTimesteps[] = {50, 150};
    const unsigned int *nextSpikeTimestep = &spikeTimesteps[0];
    const unsigned int *endSpikeTimestep = &spikeTimesteps[2];

    // Loop through timesteps
    while(t < 200.0) {
        // If there are more spikes to emit and
        // next one should be emitted this timestep
        if(nextSpikeTimestep != endSpikeTimestep
            && iT == *nextSpikeTimestep)
        {
            // Manually emit a single spike from spike source
            glbSpkCntStim[0] = 1;
            glbSpkStim[0] = 0;

            // Go onto next spike
            nextSpikeTimestep++;
        }
        else {
            glbSpkCntStim[0] = 0;
        }

        pushStimCurrentSpikesToDevice();

        // Simulate
        stepTime();

        pullVExcitatoryFromDevice();

        recorder.record(t);
    }

    return EXIT_SUCCESS;
}
