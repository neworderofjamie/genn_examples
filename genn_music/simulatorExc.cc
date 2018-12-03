#include "ModelExc_CODE/definitions.h"
#include <fstream>

#include <music.hh>

#include "../common/music.h"
#include "genn_utils/spike_csv_recorder.h"

class MyEventHandlerGlobal : public MUSIC::EventHandlerGlobalIndex 
{
public:
    void operator () (double t, MUSIC::GlobalIndex id)
    {
        // **TODO** add synaptic delay to t (time spike was GENERATED) and add into correct bin
        // Print incoming event
        spike_Inh[spikeCount_Inh]= id;
        spikeCount_Inh++;
    }
};


using namespace BoBRobotics;

int main(int argc, char *argv[])
{
    // Get real argc and argv
    // **YUCK** MUSIC::Runtime deletes this so it needs to be created as raw pointer on heap
    auto *setup = new MUSIC::Setup(argc, argv);

    // Publish an input port
    MUSIC::EventInputPort* in = setup->publishEventInput("in");

    MyEventHandlerGlobal evhandlerGlobal;  

    allocateMem();
    std::cout << "Initialising" << std::endl;
    initialize();
    initModelExc();

    MUSIC::LinearIndex indicesInh (0, 2000);
    in->map (&indicesInh, &evhandlerGlobal, 0.001, 1);

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
        
        
        spikeCount_Inh = 0;
        runtime.tick ();
        recorder.record(t);
        
    }
    runtime.finalize();

    return EXIT_SUCCESS;
}
