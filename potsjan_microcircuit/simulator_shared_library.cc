// Standard C++ includes
#include <memory>
#include <random>
#include <vector>

// GeNN robotics includes
#include "common/timer.h"
#include "genn_utils/connectors.h"
#include "genn_utils/spike_csv_recorder.h"

// Common includes
#include "../common/shared_library_model.h"

// Model parameters
#include "parameters.h"

using namespace BoBRobotics;

int main()
{
    SharedLibraryModelFloat model("potjans_microcircuit");
    {
        Timer<> timer("Allocation:");

        model.allocateMem();
    }
    {
        Timer<> timer("Initialization:");
        model.initialize();
    }

    {
        Timer<> timer("Building connectivity:");
        std::mt19937 rng;
        for(unsigned int trgLayer = 0; trgLayer < Parameters::LayerMax; trgLayer++) {
            for(unsigned int trgPop = 0; trgPop < Parameters::PopulationMax; trgPop++) {
                const std::string trgName = Parameters::getPopulationName(trgLayer, trgPop);
                const unsigned int numTrg = Parameters::getScaledNumNeurons(trgLayer, trgPop);

                // Loop through source populations and layers
                for(unsigned int srcLayer = 0; srcLayer < Parameters::LayerMax; srcLayer++) {
                    for(unsigned int srcPop = 0; srcPop < Parameters::PopulationMax; srcPop++) {
                        const std::string srcName = Parameters::getPopulationName(srcLayer, srcPop);
                        const unsigned int numSrc = Parameters::getScaledNumNeurons(srcLayer, srcPop);

                        // Find sparse projection structure and allocate function associated with projection
                        RaggedProjection<unsigned int> *raggedProjection = (RaggedProjection<unsigned int>*)model.getSymbol("C" + srcName + "_" + trgName, true);
                        if(raggedProjection) {
                            GeNNUtils::buildFixedNumberTotalWithReplacementConnector(numSrc, numTrg, Parameters::getScaledNumConnections(srcLayer, srcPop, trgLayer, trgPop),
                                                                                     *raggedProjection, rng);
                        }
                    }
                }
            }

        }
    }

    // Final setup
    {
        Timer<> timer("Sparse init:");
        model.initializeSparse();
    }

    // Create spike recorders
    std::vector<std::unique_ptr<GeNNUtils::SpikeRecorder>> spikeRecorders;
    spikeRecorders.reserve(Parameters::LayerMax * Parameters::PopulationMax);
#ifndef CPU_ONLY
    std::vector<SharedLibraryModelFloat::VoidFunction> pullCurrentSpikesFunctions;
    pullCurrentSpikesFunctions.reserve(Parameters::LayerMax * Parameters::PopulationMax);
#endif
    for(unsigned int layer = 0; layer < Parameters::LayerMax; layer++) {
        for(unsigned int pop = 0; pop < Parameters::PopulationMax; pop++) {
            const std::string name = Parameters::getPopulationName(layer, pop);

            // Get spike pull function
#ifndef CPU_ONLY
            pullCurrentSpikesFunctions.push_back(
                (SharedLibraryModelFloat::VoidFunction)model.getSymbol("pull" + name + "CurrentSpikesFromDevice"));
#endif
            // Get number of neurons in population, spike count and spike arrays
            const unsigned int numNeurons = Parameters::getScaledNumNeurons(layer, pop);
            unsigned int **spikeCount = (unsigned int**)model.getSymbol("glbSpkCnt" + name);
            unsigned int **spikes = (unsigned int**)model.getSymbol("glbSpk" + name);

            // Add cached recorder
            spikeRecorders.emplace_back(
                new GeNNUtils::SpikeCSVRecorderCached((name + ".csv").c_str(), *spikeCount, *spikes));
        }
    }

    double recordMs = 0.0;
    {
        Timer<> timer("Simulation:");
        // Loop through timesteps
        const unsigned int timesteps = round(Parameters::durationMs / Parameters::dtMs);
        const unsigned int tenPercentTimestep = timesteps / 10;
        for(unsigned int i = 0; i < timesteps; i++)
        {
            // Indicate every 10%
            if((i % tenPercentTimestep) == 0) {
                std::cout << i / 100 << "%" << std::endl;
            }

            // Simulate
#ifndef CPU_ONLY
            model.stepTimeGPU();

            // Pull current spikes from all populations
            for(auto p : pullCurrentSpikesFunctions) {
                p();
            }
#else
            model.stepTimeCPU();
#endif

            {
                TimerAccumulate<> timer(recordMs);

                // Record spikes
                for(auto &s : spikeRecorders) {
                    s->record(model.getT());
                }
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
#endif
    std::cout << "Record:" << recordMs << "ms" << std::endl;

    return 0;
}
