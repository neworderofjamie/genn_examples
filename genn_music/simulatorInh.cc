#include "ModelInh_CODE/definitions.h"
#include <fstream>

#include <music.hh>

#include "../common/music.h"
#include "genn_utils/spike_csv_recorder.h"

class MyEventHandlerGlobal : public MUSIC::EventHandlerGlobalIndex 
{
public:
    void operator () (double t, MUSIC::GlobalIndex id)
    {
        // Print incoming event
        spike_Exc[spikeCount_Exc]= id;
        spikeCount_Exc++;
    }
};
using namespace BoBRobotics;


int main(int argc, char *argv[])
{
    // **YUCK** MUSIC::Runtime deletes this so it needs to be created as raw pointer on heap
    auto *setup = new MUSIC::Setup(argc, argv);

    // Publish an input port
    MUSIC::EventInputPort* in = setup->publishEventInput ("in");

    MyEventHandlerGlobal evhandlerGlobal;  

    allocateMem();
    std::cout << "Initialising" << std::endl;
    initialize();
    initModelInh();

    MUSIC::LinearIndex indicesExc (0, 8000);
    in->map (&indicesExc, &evhandlerGlobal, 0.001, 1);

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
        spikeCount_Exc = 0;
        runtime.tick ();
        recorder.record(t);
    }
    runtime.finalize();
    return EXIT_SUCCESS;
}
