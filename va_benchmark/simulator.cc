// Standard C++ includes
#include <random>

// GeNN robotics includes
#include "common/timer.h"
#include "genn_utils/spike_csv_recorder.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "va_benchmark_CODE/definitions.h"

using namespace BoBRobotics;

int main()
{
    allocateMem();
    initialize();
    initializeSparse();

    // Open CSV output files
    GeNNUtils::SpikeCSVRecorder spikes("spikes.csv", glbSpkCntE, glbSpkE);

    while(t < 10000.0) {
        // Simulate
        stepTime();

        pullECurrentSpikesFromDevice();


        spikes.record(t);
    }

    std::cout << "Init:" << initTime << std::endl;
    std::cout << "Init sparse:" << initSparseTime << std::endl;
    std::cout << "Neuron update:" << neuronUpdateTime << std::endl;
    std::cout << "Presynaptic update:" << presynapticUpdateTime << std::endl;

    return 0;
}
