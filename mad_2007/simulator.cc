// Standard C++ includes
#include <fstream>
#include <random>

// GeNN robotics includes
#include "common/timer.h"
#include "genn_utils/spike_csv_recorder.h"
#include "third_party/path.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "mad_2007_CODE/definitions.h"

using namespace BoBRobotics;

int main(int argc, char** argv)
{
    filesystem::path outputPath;
    if(argc > 1) {
        outputPath = argv[1];
    }

    {
        Timer<> tim("Allocation:");
        allocateMem();
    }
    {
        Timer<> tim("Initialization:");
        initialize();
    }

    // Final setup
    {
        Timer<> tim("Sparse init:");
        initmad_2007();
    }

    {
        // Open CSV output files
        //GeNNUtils::SpikeCSVRecorderDelay spikes("spikes.csv", Parameters::numExcitatory,
        //                                        spkQuePtrE, glbSpkCntE, glbSpkE);
        GeNNUtils::SpikeCSVRecorderCached spikes((outputPath / "spikes.csv").str().c_str(), glbSpkCntE, glbSpkE);
        {
            Timer<> tim("Simulation:");
            // Loop through timesteps
            while(t < Parameters::durationMs) {
                if((iT % 1000) == 0) {
                    std::cout << (t / Parameters::durationMs) * 100.0 << "%" << std::endl;
                }

                // Simulate
#ifndef CPU_ONLY
                stepTimeGPU();

                pullECurrentSpikesFromDevice();
#else
                stepTimeCPU();
#endif

                spikes.record(t);
            }
        }
    }
#ifdef MEASURE_TIMING
    std::cout << "Timing:" << std::endl;
    std::cout << "\tHost init:" << initHost_tme * 1000.0 << std::endl;
    std::cout << "\tDevice init:" << initDevice_tme * 1000.0 << std::endl;
    std::cout << "\tHost sparse init:" << sparseInitHost_tme * 1000.0 << std::endl;
    std::cout << "\tDevice sparse init:" << sparseInitDevice_tme * 1000.0 << std::endl;
    std::cout << "\tNeuron simulation:" << neuron_tme * 1000.0 << std::endl;
    std::cout << "\tSynapse simulation:" << synapse_tme * 1000.0 << std::endl;
    std::cout << "\tPostsynaptic learning:" << learning_tme * 1000.0 << std::endl;
#endif
    {
        Timer<> tim("Weight analysis:");

        // Download weights
        pullEEStateFromDevice();
        
        // **HACK** Download row lengths
        extern unsigned int *d_rowLengthEE;
        CHECK_CUDA_ERRORS(cudaMemcpy(CEE.rowLength, d_rowLengthEE, Parameters::numExcitatory * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // Write row weights to file
        std::ofstream weights((outputPath / "weights.bin").str(), std::ios::binary);
        for(unsigned int i = 0; i < Parameters::numExcitatory; i++) {
            weights.write(reinterpret_cast<char*>(&gEE[i * CEE.maxRowLength]), sizeof(scalar) * CEE.rowLength[i]);
        }
    }

    return 0;
}
