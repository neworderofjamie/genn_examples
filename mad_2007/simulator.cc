// Standard C++ includes
#include <fstream>
#include <random>

// GeNN robotics includes
#include "connectors.h"
#include "spike_csv_recorder.h"
#include "timer.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "mad_2007_CODE/definitions.h"

int main()
{
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
        SpikeCSVRecorderCached spikes("spikes.csv", glbSpkCntE, glbSpkE);

        {
            Timer<> tim("Simulation:");
            // Loop through timesteps
            while(t < Parameters::durationMs) {
                if((iT % 10000) == 0) {
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
    {
        Timer<> tim("Weight analysis:");

        // Download weights
        pullEEStateFromDevice();
        
        // **HACK** Download row lengths
        extern unsigned int *d_rowLengthEE;
        CHECK_CUDA_ERRORS(cudaMemcpy(CEE.rowLength, d_rowLengthEE, Parameters::numExcitatory * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        // Write row weights to file
        std::ofstream weights("weights.bin", std::ios::binary);
        for(unsigned int i = 0; i < Parameters::numInhibitory; i++) {
            weights.write(reinterpret_cast<char*>(&gEE[i * CEE.maxRowLength]), sizeof(scalar) * CEE.rowLength[i]);
        }
    }

    return 0;
}
