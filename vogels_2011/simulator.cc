// Standard C++ includes
#include <numeric>
#include <random>

// GeNN robotics includes
#include "connectors.h"
#include "spike_csv_recorder.h"
#include "timer.h"

// Auto-generated model code
#include "vogels_2011_CODE/definitions.h"

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
#ifndef CPU_ONLY
        std::mt19937 rng;
#endif
        buildFixedProbabilityConnector(500, 500, 0.02f,
                                       CII, &allocateII, rng);
        buildFixedProbabilityConnector(500, 2000, 0.02f,
                                       CIE, &allocateIE, rng);
        buildFixedProbabilityConnector(2000, 2000, 0.02f,
                                       CEE, &allocateEE, rng);
        buildFixedProbabilityConnector(2000, 500, 0.02f,
                                       CEI, &allocateEI, rng);
    }

    // Final setup
    {
        Timer<> t("Sparse init:");
        initvogels_2011();
    }

    // Open CSV output files
    SpikeCSVRecorder spikes("spikes.csv", glbSpkCntE, glbSpkE);

    FILE *weights = fopen("weights.csv", "w");
    fprintf(weights, "Time(ms), Weight (nA)\n");

    {
        Timer<> t("Simulation:");
        // Loop through timesteps
        for(unsigned int t = 0; t < 10000; t++)
        {
            // Simulate
#ifndef CPU_ONLY
            stepTimeGPU();

            pullECurrentSpikesFromDevice();
            //pullIEStateFromDevice();
#else
            stepTimeCPU();
#endif

            spikes.record(t);


            // Calculate mean IE weights
            float totalWeight = std::accumulate(&gIE[0], &gIE[CIE.connN], 0.0f);
            fprintf(weights, "%f, %f\n", 1.0 * (double)t, totalWeight / (double)CIE.connN);

        }
    }
    
    std::cout << "Timing:" << std::endl;
    std::cout << "\tHost init:" << initHost_tme * 1000.0 << std::endl;
    std::cout << "\tDevice init:" << initDevice_tme * 1000.0 << std::endl;
    std::cout << "\tHost sparse init:" << sparseInitHost_tme * 1000.0 << std::endl;
    std::cout << "\tDevice sparse init:" << sparseInitDevice_tme * 1000.0 << std::endl;
    std::cout << "\tNeuron simulation:" << neuron_tme * 1000.0 << std::endl;
    std::cout << "\tSynapse simulation:" << synapse_tme * 1000.0 << std::endl;
    std::cout << "\tPost learning similation:" << learning_tme * 1000.0 << std::endl;
    // Close files
    fclose(weights);

    return 0;
}
