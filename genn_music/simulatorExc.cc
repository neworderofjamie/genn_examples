#include "ModelExc_CODE/definitions.h"
#include <fstream>

#include <music.hh>

#include "../common/music.h"
#include "genn_utils/spike_csv_recorder.h"

using namespace BoBRobotics;

int main(int argc, char *argv[])
{
    // Get real argc and argv
    // **YUCK** MUSIC::Runtime deletes this so it needs to be created as raw pointer on heap
    auto *setup = new MUSIC::Setup(argc, argv);

    allocateMem();
    std::cout << "Initialising" << std::endl;
    initialize();
    initModelExc();

    MUSICSpikeIn spikeIn("in", 2000, DT, glbSpkCntInh, glbSpkInh, setup);
    MUSICSpikeOut spikeOut("out", 8000, glbSpkCntExc, glbSpkExc, setup);

    // Prepare for simulation
    MUSIC::Runtime runtime(setup, 0.001);

    std::cout << "Simulating" << std::endl;
    BoBRobotics::GeNNUtils::SpikeCSVRecorder recorder("SpikesInh.csv", glbSpkCntInh, glbSpkInh);
    while(t < 1000.0f) {
#ifdef CPU_ONLY
        stepTimeCPU();
#else
        pushInhCurrentSpikesToDevice();
        stepTimeGPU();
        pullExcCurrentSpikesFromDevice();
        pullInhCurrentSpikesFromDevice();
#endif

        spikeOut.transmit(t);
        spikeIn.tick();

        runtime.tick ();

        recorder.record(t);
        
    }
    runtime.finalize();

    return EXIT_SUCCESS;
}
