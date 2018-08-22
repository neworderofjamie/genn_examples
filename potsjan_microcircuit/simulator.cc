// Standard C++ includes
#include <memory>
#include <random>
#include <vector>

// GeNN robotics includes
#include "common/timer.h"
#include "genn_utils/connectors.h"
#include "genn_utils/spike_csv_recorder.h"

// Model parameters
#include "parameters.h"

// Auto-generated model code
#include "potjans_microcircuit_CODE/definitions.h"

// Macro to build a connection between a pair of populations
#define BUILD_PROJECTION(SRC_LAYER, SRC_POP, TRG_LAYER, TRG_POP)                                                                                                    \
    GeNNUtils::buildFixedNumberTotalWithReplacementConnector(Parameters::getScaledNumNeurons(Parameters::Layer##SRC_LAYER, Parameters::Population##SRC_POP),        \
                                                             Parameters::getScaledNumNeurons(Parameters::Layer##TRG_LAYER, Parameters::Population##TRG_POP),        \
                                                             Parameters::getScaledNumConnections(Parameters::Layer##SRC_LAYER, Parameters::Population##SRC_POP,     \
                                                                                                 Parameters::Layer##TRG_LAYER, Parameters::Population##TRG_POP),    \
                                                             C##SRC_LAYER##SRC_POP##_##TRG_LAYER##TRG_POP, rng)

// Macro to record a population's output
#define ADD_SPIKE_RECORDER(LAYER, POPULATION)                                                                                                               \
    spikeRecorders.emplace_back(new GeNNUtils::SpikeCSVRecorderCached(#LAYER#POPULATION".csv", glbSpkCnt##LAYER##POPULATION, glbSpk##LAYER##POPULATION))


    using namespace BoBRobotics;

int main()
{
    {
        Timer<> timer("Allocation:");
        allocateMem();
    }
    {
        Timer<> timer("Initialization:");
        initialize();
    }

    {
        Timer<> timer("Building connectivity:");

        std::mt19937 rng;
        BUILD_PROJECTION(23, E, 23, E);
        BUILD_PROJECTION(23, E, 23, I);
        BUILD_PROJECTION(23, E, 4, E);
        BUILD_PROJECTION(23, E, 4, I);
        BUILD_PROJECTION(23, E, 5, E);
        BUILD_PROJECTION(23, E, 5, I);
        BUILD_PROJECTION(23, E, 6, E);
        BUILD_PROJECTION(23, E, 6, I);
        BUILD_PROJECTION(23, I, 23, E);
        BUILD_PROJECTION(23, I, 23, I);
        BUILD_PROJECTION(23, I, 4, E);
        BUILD_PROJECTION(23, I, 4, I);
        BUILD_PROJECTION(23, I, 5, E);
        BUILD_PROJECTION(23, I, 5, I);
        BUILD_PROJECTION(23, I, 6, E);
        BUILD_PROJECTION(23, I, 6, I);
        BUILD_PROJECTION(4, E, 23, E);
        BUILD_PROJECTION(4, E, 23, I);
        BUILD_PROJECTION(4, E, 4, E);
        BUILD_PROJECTION(4, E, 4, I);
        BUILD_PROJECTION(4, E, 5, E);
        BUILD_PROJECTION(4, E, 5, I);
        BUILD_PROJECTION(4, E, 6, E);
        BUILD_PROJECTION(4, E, 6, I);
        BUILD_PROJECTION(4, I, 23, E);
        BUILD_PROJECTION(4, I, 23, I);
        BUILD_PROJECTION(4, I, 4, E);
        BUILD_PROJECTION(4, I, 4, I);
        BUILD_PROJECTION(4, I, 5, E);
        BUILD_PROJECTION(4, I, 5, I);
        BUILD_PROJECTION(4, I, 6, E);
        BUILD_PROJECTION(4, I, 6, I);
        BUILD_PROJECTION(5, E, 23, E);
        BUILD_PROJECTION(5, E, 23, I);
        BUILD_PROJECTION(5, E, 4, E);
        BUILD_PROJECTION(5, E, 4, I);
        BUILD_PROJECTION(5, E, 5, E);
        BUILD_PROJECTION(5, E, 5, I);
        BUILD_PROJECTION(5, E, 6, E);
        BUILD_PROJECTION(5, E, 6, I);
        //BUILD_PROJECTION(5, I, 23, E);
        //BUILD_PROJECTION(5, I, 23, I);
        BUILD_PROJECTION(5, I, 4, E);
        //BUILD_PROJECTION(5, I, 4, I);
        BUILD_PROJECTION(5, I, 5, E);
        BUILD_PROJECTION(5, I, 5, I);
        BUILD_PROJECTION(5, I, 6, E);
        BUILD_PROJECTION(5, I, 6, I);
        BUILD_PROJECTION(6, E, 23, E);
        BUILD_PROJECTION(6, E, 23, I);
        BUILD_PROJECTION(6, E, 4, E);
        BUILD_PROJECTION(6, E, 4, I);
        BUILD_PROJECTION(6, E, 5, E);
        BUILD_PROJECTION(6, E, 5, I);
        BUILD_PROJECTION(6, E, 6, E);
        BUILD_PROJECTION(6, E, 6, I);
        //BUILD_PROJECTION(6, I, 23, E);
        //BUILD_PROJECTION(6, I, 23, I);
        //BUILD_PROJECTION(6, I, 4, E);
        //BUILD_PROJECTION(6, I, 4, I);
        //BUILD_PROJECTION(6, I, 5, E);
        //BUILD_PROJECTION(6, I, 5, I);
        BUILD_PROJECTION(6, I, 6, E);
        BUILD_PROJECTION(6, I, 6, I);
    }

    // Final setup
    {
        Timer<> timer("Sparse init:");
        initpotjans_microcircuit();
    }

    // Create spike recorders
    // **HACK** would be nicer to have arrays of objects rather than pointers but ofstreams
    // aren't correctly moved in GCC 4.9.4 (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54316) -
    // the newest version that can be used with CUDA on Sussex HPC
    std::vector<std::unique_ptr<GeNNUtils::SpikeCSVRecorderCached>> spikeRecorders;
    spikeRecorders.reserve(Parameters::LayerMax * Parameters::PopulationMax);
    ADD_SPIKE_RECORDER(23, E);
    ADD_SPIKE_RECORDER(23, I);
    ADD_SPIKE_RECORDER(4, E);
    ADD_SPIKE_RECORDER(4, I);
    ADD_SPIKE_RECORDER(5, E);
    ADD_SPIKE_RECORDER(5, I);
    ADD_SPIKE_RECORDER(6, E);
    ADD_SPIKE_RECORDER(6, I);

    double recordMs = 0.0;

    {
        Timer<> timer("Simulation:");
        // Loop through timesteps
        const unsigned int timesteps = round(Parameters::durationMs / DT);
        const unsigned int tenPercentTimestep = timesteps / 10;
        for(unsigned int i = 0; i < timesteps; i++)
        {
            // Indicate every 10%
            if((i % tenPercentTimestep) == 0) {
                std::cout << i / 100 << "%" << std::endl;
            }

            // Simulate
#ifndef CPU_ONLY
            stepTimeGPU();

            pull23ECurrentSpikesFromDevice();
            pull23ICurrentSpikesFromDevice();
            pull4ECurrentSpikesFromDevice();
            pull4ICurrentSpikesFromDevice();
            pull5ECurrentSpikesFromDevice();
            pull5ICurrentSpikesFromDevice();
            pull6ECurrentSpikesFromDevice();
            pull6ICurrentSpikesFromDevice();
#else
            stepTimeCPU();
#endif

            {
                TimerAccumulate<> timer(recordMs);

                // Record spikes
                for(auto &s : spikeRecorders) {
                    s->record(t);
                }
            }
        }
    }

    // Write spike recorder cache to disk
    {
        Timer<> timer("Writing spikes to disk:");
        for(auto &s : spikeRecorders) {
            s->writeCache();
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
#endif
    std::cout << "Record:" << recordMs << "ms" << std::endl;

    return 0;
}
