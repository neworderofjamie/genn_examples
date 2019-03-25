#include "if_curr_CODE/definitions.h"

int main()
{
    allocateMem();
    initialize();
    initializeSparse();

    FILE *membraneVoltage = fopen("voltages.csv", "w");
    fprintf(membraneVoltage, "Time(ms), Voltage (mV)\n");

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
        // Calculate simulation time
        const double time = 1.0 * (double)t;
        fprintf(membraneVoltage, "%f, %f, %f\n", time, VExcitatory[0], inSynStimToExcitatory[0]);

    }

    fclose(membraneVoltage);

    return EXIT_SUCCESS;
}
