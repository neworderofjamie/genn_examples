// Standard C++ includes
#include <fstream>
#include <random>

// GeNN user project includes
#include "timer.h"
#include "spikeRecorder.h"
//#include "third_party/path.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "mad_2007_CODE/definitions.h"

int main(int argc, char** argv)
{
    std::string outputPath;
    if(argc > 1) {
        outputPath = argv[1];
    }

    allocateMem();
    initialize();
    initializeSparse();
    {
        // Open CSV output files
        SpikeRecorderDelayCached spikes("spikes.csv", Parameters::numExcitatory,
                                        spkQuePtrE, glbSpkCntE, glbSpkE, ",", true);
        {
            Timer tim("Simulation:");
            // Loop through timesteps
            double averageSpikes = 0.0;
            const double alpha = 0.001;
            while(t < Parameters::durationMs) {
                // Simulate
                stepTime();
                pullECurrentSpikesFromDevice();

                averageSpikes = (alpha * (double)spikeCount_E) + ((1.0 - alpha) * averageSpikes);
                if((iT % 1000) == 0) {

                    std::cout << (t / Parameters::durationMs) * 100.0 << "%" << std::endl;
                    std::cout << "Moving average spike rate:" << (averageSpikes / (double)Parameters::numExcitatory) / (Parameters::timestep / 1000.0) << " Hz" << std::endl;
                }

                // Record last 50s of spiking activity
                //if(t > (Parameters::durationMs - (50.0 * 1000.0))) {
                    spikes.record(t);
                //}
            }
        }
    }

    if(Parameters::measureTiming) {
        std::cout << "Timing:" << std::endl;
        std::cout << "\tInit:" << initTime * 1000.0 << std::endl;
        std::cout << "\tSparse init:" << initSparseTime * 1000.0 << std::endl;
        std::cout << "\tNeuron simulation:" << neuronUpdateTime * 1000.0 << std::endl;
        std::cout << "\tSynapse simulation:" << presynapticUpdateTime * 1000.0 << std::endl;
#ifndef STATIC
        std::cout << "\tPostsynaptic learning:" << postsynapticUpdateTime * 1000.0 << std::endl;
#endif
    }

#ifndef STATIC
    {
        Timer tim("Weight analysis:");

        // Download weights and connectivity
        pullEEStateFromDevice();
        pullEEConnectivityFromDevice();

        // Write row weights to file
        std::ofstream weights(outputPath + "/weights.bin", std::ios::binary);
        for(unsigned int i = 0; i < Parameters::numExcitatory; i++) {
            weights.write(reinterpret_cast<char*>(&gEE[i * maxRowLengthEE]), sizeof(scalar) * rowLengthEE[i]);
        }
    }
#endif // !STATIC
    return EXIT_SUCCESS;
}
