#include "model.cc"
#include "dendritic_delay_CODE/definitions.h"

#include "genn_utils/spike_csv_recorder.h"

using namespace BoBRobotics;

int main()
{
    allocateMem();
    initialize();

    for(unsigned int i = 0; i < 10; i++) {
        dStimToExcitatory[i] = i / 2;
    }

    initdendritic_delay();

    // Open CSV output files
    GeNNUtils::SpikeCSVRecorder spikes("spikes.csv", glbSpkCntExcitatory, glbSpkExcitatory);


    // Loop through timesteps
    for(unsigned int i = 0; i < 200; i++)
    {
        if(i == 0) {
            // Manually emit a single spike from spike source
            glbSpkCntStim[0] = 1;
            glbSpkStim[0] = 0;
        }
        else {
            glbSpkCntStim[0] = 0;
        }
#ifdef CPU_ONLY
        // Simulate
        stepTimeCPU();
#else
        pushStimCurrentSpikesToDevice();

        stepTimeGPU();

        pullExcitatoryCurrentSpikesFromDevice();
#endif
        spikes.record(t);
    }


    return 0;
}