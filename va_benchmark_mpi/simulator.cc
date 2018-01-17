// Standard C++ includes
#include <random>

// GeNN robotics includes
#include "connectors.h"
#include "spike_csv_recorder.h"
#include "timer.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#ifdef DEFINITIONS_HEADER
#include DEFINITIONS_HEADER
#else
#include "va_benchmark_CODE/definitions.h"
#endif

int main()
{
    {
        Timer<> t("Allocation:");
        allocateMem();
    }
    {
        Timer<> t("Initialization:");
        initialize();
    }

    {
        Timer<> t("Building connectivity:");

        std::mt19937 rng;

        // If the inhibitory population is being simulated on the local machine build its afferent connectivity
#ifndef I_REMOTE
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                       CII, &allocateII, rng);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                       CEI, &allocateEI, rng);
#endif  // I_REMOTE

        // If the excitatory population is being simulated on the local machine build its afferent connectivity
#ifndef E_REMOTE
        buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                       CIE, &allocateIE, rng);
        buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                       CEE, &allocateEE, rng);
#endif  // E_REMOTE
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initva_benchmark();
    }

    // Open CSV output files for populations being simulated on local machine
#ifndef E_REMOTE
    SpikeCSVRecorder spikesE("spikes_e.csv", glbSpkCntE, glbSpkE);
#endif  // E_REMOTE
#ifndef I_REMOTE
    SpikeCSVRecorder spikesI("spikes_i.csv", glbSpkCntI, glbSpkI);
#endif  // E_REMOTE
    {
        Timer<> t("Simulation:");
        // Loop through timesteps
        for(unsigned int t = 0; t < 10000; t++)
        {
            // Simulate
#ifndef CPU_ONLY
            stepTimeGPU();

            // Pull spikes to host for populations being simulated on local machine
#ifndef E_REMOTE
            pullECurrentSpikesFromDevice();
#endif  // E_REMOTE
#ifndef I_REMOTE
            pullICurrentSpikesFromDevice();
#endif  // I_REMOTE
#else
            stepTimeCPU();
#endif
            // Record spikes to disk for populations being simulated on local machine
#ifndef E_REMOTE
            spikesE.record(t);
#endif  // E_REMOTE
#ifndef I_REMOTE
            spikesI.record(t);
#endif  // I_REMOTE

#ifdef MPI_ENABLE
            // Synchronise nodes using MPI
            communicateSpikes();

            // Push spikes received from remote populations to device
#ifdef E_REMOTE
            pushECurrentSpikesToDevice();
#endif  // E_REMOTE
#ifdef I_REMOTE
            pushICurrentSpikesToDevice();
#endif  // I_REMOTE
#endif  // MPI_ENABLE
        }
    }

    // Exit GeNN
    // **NOTE** this is particularily important for MPI simulations as MPI_Finalize is called here
    exitGeNN();
    return 0;
}
