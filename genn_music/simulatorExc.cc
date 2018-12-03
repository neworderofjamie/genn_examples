#include "ExcModel_CODE/definitions.h"
#include <fstream>

#include <music.hh>

#include "../common/music.h"

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
    initExcModel();

    MUSIC::LinearIndex indicesInh (0, 2000);
    in->map (&indicesInh, &evhandlerGlobal, 0.001, 1);

    MUSICSpikeOut spikeOut("out", 8000, glbSpkCntExc, glbSpkExc, setup);

    std::cout << "Created MUSICSpikeOut1" << std::endl;
    // Prepare for simulation
    MUSIC::Runtime runtime(setup, 0.001);

    std::cout << "Simulating" << std::endl;
    std::ofstream stream("ExcSpikes1.csv");
    std::ofstream streamInh("InhSpikes1.csv");
    while(t < 1000.0f) {
#ifdef CPU_ONLY
        stepTimeCPU();
#else
        pushInhCurrentSpikesToDevice();
        stepTimeGPU();
        pullExcCurrentSpikesFromDevice();
        pullInhCurrentSpikesFromDevice();
#endif

        spikeOut.record(t);
        
        
        spikeCount_Inh = 0;
        runtime.tick ();
        for(unsigned int i = 0; i < spikeCount_Inh; i++) {
            streamInh << t << ", " << spike_Inh[i] << std::endl;
        }
        
    }
    runtime.finalize();

    return EXIT_SUCCESS;
}
