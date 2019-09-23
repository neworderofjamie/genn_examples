// Standard C++ includes
#include <memory>
#include <random>
#include <vector>

// GeNN user project includes
#include "sharedLibraryModel.h"
#include "spikeRecorder.h"
#include "timer.h"

// Model parameters
#include "parameters.h"
#include "utils.h"

int main()
{
    SharedLibraryModel<float> model("./", "potjans_microcircuit");

    model.allocateMem();

    {
        Timer timer("Building row lengths:");

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

                        // If synapse group exists
                        const std::string synapsePopName = srcName + "_" + trgName;
                        void *rowLengths = model.getSymbol("preCalcRowLength" + synapsePopName, true);
                        if(rowLengths) {
                            // Allocate row lengths
                            model.allocateExtraGlobalParam(synapsePopName, "preCalcRowLength", numSrc);

                            // Build row lengths on host
                            buildRowLengths(numSrc, numTrg, Parameters::getScaledNumConnections(srcLayer, srcPop, trgLayer, trgPop),
                                            *(static_cast<unsigned int**>(rowLengths)), rng);

                            // Push to device
                            model.pushExtraGlobalParam(synapsePopName, "preCalcRowLength", numSrc);
                        }
                    }
                }
            }
        }
    }

    model.initialize();
    model.initializeSparse();

    // Create spike recorders
    std::vector<std::unique_ptr<SpikeRecorderCached>> spikeRecorders;
    spikeRecorders.reserve(Parameters::LayerMax * Parameters::PopulationMax);
    for(unsigned int layer = 0; layer < Parameters::LayerMax; layer++) {
        for(unsigned int pop = 0; pop < Parameters::PopulationMax; pop++) {
            const std::string name = Parameters::getPopulationName(layer, pop);

            // Get spike count and spike arrays
            unsigned int *spikeCount = model.getArray<unsigned int>("glbSpkCnt" + name);
            unsigned int *spikes = model.getArray<unsigned int>("glbSpk" + name);

            // Add cached recorder
            spikeRecorders.emplace_back(
                new SpikeRecorderCached((name + ".csv").c_str(), spikeCount, spikes, ",", true));
        }
    }

    double recordS = 0.0;
    {
        Timer timer("Simulation:");
        // Loop through timesteps
        const unsigned int timesteps = round(Parameters::durationMs / Parameters::dtMs);
        const unsigned int tenPercentTimestep = timesteps / 10;
        for(unsigned int i = 0; i < timesteps; i++) {
            // Indicate every 10%
            if((i % tenPercentTimestep) == 0) {
                std::cout << i / 100 << "%" << std::endl;
            }

            // Simulate
            model.stepTime();

            // Pull spikes from each population from device
            for(unsigned int layer = 0; layer < Parameters::LayerMax; layer++) {
                for(unsigned int pop = 0; pop < Parameters::PopulationMax; pop++) {
                    model.pullCurrentSpikesFromDevice(Parameters::getPopulationName(layer, pop));
                }
            }

            {
                TimerAccumulate timer(recordS);

                // Record spikes
                for(auto &s : spikeRecorders) {
                    s->record(model.getTime());
                }
            }
        }
    }

    if(Parameters::measureTiming) {
        std::cout << "Timing:" << std::endl;
        std::cout << "\tInit:" << *model.getScalar<double>("initTime") * 1000.0 << std::endl;
        std::cout << "\tSparse init:" << *model.getScalar<double>("initSparseTime") * 1000.0 << std::endl;
        std::cout << "\tNeuron simulation:" << *model.getScalar<double>("neuronUpdateTime") * 1000.0 << std::endl;
        std::cout << "\tSynapse simulation:" << *model.getScalar<double>("presynapticUpdateTime") * 1000.0 << std::endl;
    }
    std::cout << "Record:" << recordS << "s" << std::endl;

    return 0;
}
