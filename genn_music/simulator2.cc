#include "InhModel_CODE/definitions.h"
#include <fstream>

#include <music.hh>

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


int main(int argc, char *argv[])
{
    // Get real argc and argv
    MUSIC::Setup* setup = new MUSIC::Setup (argc, argv);

    // Publish an input port
    MUSIC::EventInputPort* in = setup->publishEventInput ("in");

    // Publish an input port
    MUSIC::EventOutputPort* out = setup->publishEventOutput ("out");

    MyEventHandlerGlobal evhandlerGlobal;  

    allocateMem();
    std::cout << "Initialising" << std::endl;
    initialize();
    initInhModel();

    MUSIC::LinearIndex indicesExc (0, 8000);
    in->map (&indicesExc, &evhandlerGlobal, 0.001, 1);

    MUSIC::LinearIndex indicesInh (0, 2000);
    out->map(&indicesInh, MUSIC::Index::GLOBAL);

    // Prepare for simulation
    MUSIC::Runtime runtime(setup, 0.001);

    std::cout << "Simulating" << std::endl;
    std::ofstream stream("InhSpikes.csv");
    while(t < 1000.0f) {
#ifdef CPU_ONLY
        stepTimeCPU();
#else
        pushExcCurrentSpikesToDevice();
        stepTimeGPU();
        pullInhCurrentSpikesFromDevice();
#endif
        for(unsigned int i = 0; i < spikeCount_Inh; i++) {
            stream << t << ", " << spike_Inh[i] << std::endl;
            out->insertEvent(t/1000.0, MUSIC::GlobalIndex(spike_Inh[i]));
        }
        spikeCount_Exc = 0;
        runtime.tick ();
    }
    runtime.finalize();
    return EXIT_SUCCESS;
}
