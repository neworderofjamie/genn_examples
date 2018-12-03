#include "ModelInh_CODE/definitions.h"
#include <fstream>

#include <music.hh>

#include "../common/music.h"
#include "genn_utils/spike_csv_recorder.h"

using namespace BoBRobotics;


int main(int argc, char *argv[])
{
    // **YUCK** MUSIC::Runtime deletes this so it needs to be created as raw pointer on heap
    auto *setup = new MUSIC::Setup(argc, argv);

    allocateMem();
    std::cout << "Initialising" << std::endl;
    initialize();
    initModelInh();

    MUSICSpikeIn spikeIn("in", 8000, DT, glbSpkCntExc, glbSpkExc, setup);
    MUSICSpikeOut spikeOut("out", 2000, glbSpkCntInh, glbSpkInh, setup);

    // Prepare for simulation
    MUSIC::Runtime runtime(setup, 0.001);

    std::cout << "Simulating" << std::endl;
    BoBRobotics::GeNNUtils::SpikeCSVRecorder recorder("SpikesExc.csv", glbSpkCntExc, glbSpkExc);
    while(t < 1000.0f) {
#ifdef CPU_ONLY
        stepTimeCPU();
#else
        pushExcCurrentSpikesToDevice();
        stepTimeGPU();
        pullInhCurrentSpikesFromDevice();
#endif
        spikeOut.transmit(t);
        spikeIn.tick();

        runtime.tick();

        recorder.record(t);
    }
    runtime.finalize();
    return EXIT_SUCCESS;
}
