// Standard C++ includes
#include <memory>
#include <random>
#include <vector>

// GeNN robotics includes
#include "connectors.h"
#include "spike_csv_recorder.h"
#include "timer.h"

// Common includes
#include "../common/shared_library_model.h"

// Model parameters
#include "parameters.h"

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
                        SparseProjection *sparseProjection = (SparseProjection*)model.getSymbol("C" + srcName + "_" + trgName, true);
                        AllocateFn allocateFn = (AllocateFn)model.getSymbol("allocate" + srcName + "_" + trgName, true);
                        if(sparseProjection && allocateFn) {
                            buildFixedNumberTotalWithReplacementConnector(numSrc, numTrg, Parameters::getScaledNumConnections(srcLayer, srcPop, trgLayer, trgPop),
                                                                          *sparseProjection, allocateFn, rng);
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
    // **HACK** would be nicer to have arrays of objects rather than pointers but ofstreams
    // aren't correctly moved in GCC 4.9.4 (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54316) -
    // the newest version that can be used with CUDA on Sussex HPC
#ifdef USE_DELAY
    std::vector<std::unique_ptr<SpikeCSVRecorderDelay>> spikeRecorders;
#else
    std::vector<std::unique_ptr<SpikeCSVRecorderCached>> spikeRecorders;
#endif
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
            // Get spike count and spike arrays
            unsigned int **spikeCount = (unsigned int**)model.getSymbol("glbSpkCnt" + name);
            unsigned int **spikes = (unsigned int**)model.getSymbol("glbSpk" + name);
#ifdef USE_DELAY
            const unsigned int numNeurons = Parameters::getScaledNumNeurons(layer, pop);
            unsigned int *spikeQueuePointer = (unsigned int*)model.getSymbol("spkQuePtr" + name);
            spikeRecorders.emplace_back(
                new SpikeCSVRecorderDelay((name + ".csv").c_str(), numNeurons,
                                          *spikeQueuePointer, *spikeCount, *spikes));
#else
            spikeRecorders.emplace_back(
                new SpikeCSVRecorderCached((name + ".csv").c_str(), *spikeCount, *spikes));
#endif
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

    // Write spike recorder cache to disk
    for(auto &s : spikeRecorders) {
        s->writeCache();
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
