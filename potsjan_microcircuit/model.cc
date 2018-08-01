
// GeNN includes
#include "modelSpec.h"

// GeNN robotics includes
#include "genn_models/exp_curr.h"
#include "genn_models/lif.h"
#include "genn_utils/connectors.h"

// Genn examples includes
#include "../common/normal_distribution.h"
#include "../common/poisson_current_source.h"

// Model includes
#include "parameters.h"

using namespace BoBRobotics;

//----------------------------------------------------------------------------
// NormalClipped
//----------------------------------------------------------------------------
class NormalClipped : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(NormalClipped, 4);

    SET_CODE(
        "scalar normal;\n"
        "do\n"
        "{\n"
        "   normal = $(mean) + ($(gennrand_normal) * $(sd));\n"
        "} while (normal > $(max) || normal < $(min));\n"
        "$(value) = normal;\n");

    SET_PARAM_NAMES({"mean", "sd", "min", "max"});
};
IMPLEMENT_SNIPPET(NormalClipped);

//----------------------------------------------------------------------------
// NormalClippedDelay
//----------------------------------------------------------------------------
class NormalClippedDelay : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(NormalClippedDelay, 4);

    SET_CODE(
        "scalar normal;\n"
        "do\n"
        "{\n"
        "   normal = $(mean) + ($(gennrand_normal) * $(sd));\n"
        "} while (normal > $(max) || normal < $(min));\n"
        "$(value) = (uint8_t)rint(normal / DT);\n");

    SET_PARAM_NAMES({"mean", "sd", "min", "max"});
};
IMPLEMENT_SNIPPET(NormalClippedDelay);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(Parameters::dtMs);
    model.setName("potjans_microcircuit");
#ifdef MEASURE_TIMING
    model.setTiming(true);
#endif

    GENN_PREFERENCES::buildSharedLibrary = false;
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::optimizeCode = true;
    GENN_PREFERENCES::mergePostsynapticModels = true;

    InitVarSnippet::Normal::ParamValues vDist(
        -58.0, // 0 - mean
        5.0);  // 1 - sd

    // LIF initial conditions
    GeNNModels::LIF::VarValues lifInit(
        initVar<InitVarSnippet::Normal>(vDist), // 0 - V
        0.0);                                   // 1 - RefracTime

    // Current source initial conditions
    PoissonCurrentSourceExp::VarValues poissonCurrInit(
        0.0);                                   // 0 - iPoisson

    // Exponential current parameters
    GeNNModels::ExpCurr::ParamValues excitatoryExpCurrParams(
        0.5);  // 0 - TauSyn (ms)

    GeNNModels::ExpCurr::ParamValues inhibitoryExpCurrParams(
        0.5);  // 0 - TauSyn (ms)

    const double quantile = 0.9999;
    const double maxDelayMs[Parameters::PopulationMax] = {
        Parameters::meanDelay[Parameters::PopulationE] + (Parameters::delaySD[Parameters::PopulationE] * normalCDFInverse(quantile)),
        Parameters::meanDelay[Parameters::PopulationI] + (Parameters::delaySD[Parameters::PopulationI] * normalCDFInverse(quantile))};
    std::cout << "Max excitatory delay: " << maxDelayMs[Parameters::PopulationE] << "ms, max inhibitory delay: " << maxDelayMs[Parameters::PopulationI] << "ms" << std::endl;

    // Calculate maximum dendritic delay slots
    // **NOTE** it seems inefficient using maximum for all but this allows more aggressive merging of postsynaptic models
    const unsigned int maxDendriticDelaySlots = (unsigned int)std::rint(std::max(maxDelayMs[Parameters::PopulationE], maxDelayMs[Parameters::PopulationI])  / Parameters::dtMs);
    std::cout << "Max dendritic delay slots:" << maxDendriticDelaySlots << std::endl;

    // Loop through populations and layers
    std::cout << "Creating neuron populations:" << std::endl;
    unsigned int totalNeurons = 0;
    for(unsigned int layer = 0; layer < Parameters::LayerMax; layer++) {
        for(unsigned int pop = 0; pop < Parameters::PopulationMax; pop++) {
            // Determine name of population
            const std::string popName = Parameters::getPopulationName(layer, pop);

            // Calculate external input rate, weight and current
            const double extInputRate = (Parameters::numExternalInputs[layer][pop] *
                                         Parameters::connectivityScalingFactor *
                                         Parameters::backgroundRate);
            const double extWeight = Parameters::externalW / sqrt(Parameters::connectivityScalingFactor);

            const double extInputCurrent = 0.001 * 0.5 * (1.0 - sqrt(Parameters::connectivityScalingFactor)) * Parameters::getFullMeanInputCurrent(layer, pop);
            assert(extInputCurrent >= 0.0);

            // LIF model parameters
            GeNNModels::LIF::ParamValues lifParams(
                0.25,               // 0 - C
                10.0,               // 1 - TauM
                -65.0,              // 2 - Vrest
                -65.0,              // 3 - Vreset
                -50.0,              // 4 - Vthresh
                extInputCurrent,    // 5 - Ioffset
                2.0);               // 6 - TauRefrac

            PoissonCurrentSourceExp::ParamValues poissonCurrParams(
                extInputRate,       // 0 - rate
                extWeight,          // 1 - weight
                0.5);               // 2 - tauSyn

            // Create population
            const unsigned int popSize = Parameters::getScaledNumNeurons(layer, pop);
            auto *neuronPop = model.addNeuronPopulation<GeNNModels::LIF>(popName, popSize,
                                                                         lifParams, lifInit);

            // Create current source
            model.addCurrentSource<PoissonCurrentSourceExp>(popName + "_curr", popName,
                                                            poissonCurrParams, poissonCurrInit);
            // Make spikes recordable on host
#ifdef USE_ZERO_COPY
            neuronPop->setSpikeVarMode(VarMode::LOC_ZERO_COPY_INIT_DEVICE);
#else
            neuronPop->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
#endif
            std::cout << "\tPopulation " << popName << ": num neurons:" << popSize << ", external input rate:" << extInputRate << ", external weight:" << extWeight << ", external DC offset:" << extInputCurrent << std::endl;
            // Add number of neurons to total
            totalNeurons += popSize;
        }
    }

    // Loop through target populations and layers
    std::cout << "Creating synapse populations:" << std::endl;
    unsigned int totalSynapses = 0;
    for(unsigned int trgLayer = 0; trgLayer < Parameters::LayerMax; trgLayer++) {
        for(unsigned int trgPop = 0; trgPop < Parameters::PopulationMax; trgPop++) {
            const std::string trgName = Parameters::getPopulationName(trgLayer, trgPop);
            const unsigned int numTrg = Parameters::getScaledNumNeurons(trgLayer, trgPop);

            // Loop through source populations and layers
            for(unsigned int srcLayer = 0; srcLayer < Parameters::LayerMax; srcLayer++) {
                for(unsigned int srcPop = 0; srcPop < Parameters::PopulationMax; srcPop++) {
                    const std::string srcName = Parameters::getPopulationName(srcLayer, srcPop);
                    const unsigned int numSrc = Parameters::getScaledNumNeurons(srcLayer, srcPop);

                    // Determine mean weight
                    const double meanWeight = Parameters::getMeanWeight(srcLayer, srcPop, trgLayer, trgPop) / sqrt(Parameters::connectivityScalingFactor);

                    // Determine weight standard deviation
                    double weightSD;
                    if(srcPop == Parameters::PopulationE && srcLayer == Parameters::Layer4 && trgLayer == Parameters::Layer23 && trgPop == Parameters::PopulationE) {
                        weightSD = meanWeight * Parameters::layer234RelW;
                    }
                    else {
                        weightSD = fabs(meanWeight * Parameters::relW);
                    }

                    // Calculate number of connections
                    const unsigned int numConnections = Parameters::getScaledNumConnections(srcLayer, srcPop, trgLayer, trgPop);

                    if(numConnections > 0) {
                        std::cout << "\tConnection between '" << srcName << "' and '" << trgName << "': numConnections=" << numConnections << ", meanWeight=" << meanWeight << ", weightSD=" << weightSD << ", meanDelay=" << Parameters::meanDelay[srcPop] << ", delaySD=" << Parameters::delaySD[srcPop] << std::endl;

                        totalSynapses += numConnections;

                        // Build unique synapse name
                        const std::string synapseName = srcName + "_" + trgName;

                        // Excitatory
                        if(srcPop == Parameters::PopulationE) {
                            // Build distribution for weight parameters
                            NormalClipped::ParamValues wDist(
                                meanWeight,                                 // 0 - mean
                                weightSD,                                   // 1 - sd
                                0.0,                                        // 2 - min
                                std::numeric_limits<float>::max());         // 3 - max

                            // Build distribution for delay parameters
                            NormalClippedDelay::ParamValues dDist(
                                Parameters::meanDelay[srcPop],              // 0 - mean
                                Parameters::delaySD[srcPop],                // 1 - sd
                                0.0,                                        // 2 - min
                                maxDelayMs[srcPop]);                        // 3 - max

                            // Create weight parameters
                            WeightUpdateModels::StaticPulseDendriticDelay::VarValues staticSynapseInit(
                                initVar<NormalClipped>(wDist),          // 0 - Wij (nA)
                                initVar<NormalClippedDelay>(dDist));    // 1 - delay (ms)

                            // Add synapse population
                            auto *synPop = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, GeNNModels::ExpCurr>(
                                synapseName, SYNAPSE_MATRIX_TYPE, NO_DELAY,
                                srcName, trgName,
                                {}, staticSynapseInit,
                                excitatoryExpCurrParams, {});

                            // Set max connections
                            synPop->setMaxConnections(
                                GeNNUtils::calcFixedNumberTotalWithReplacementConnectorMaxConnections(numSrc, numTrg, numConnections));

                            synPop->setMaxDendriticDelayTimesteps(maxDendriticDelaySlots);
                        }
                        // Inhibitory
                        else {
                            // Build distribution for weight parameters
                            NormalClipped::ParamValues wDist(
                                meanWeight,                                 // 0 - mean
                                weightSD,                                   // 1 - sd
                                -std::numeric_limits<float>::max(),         // 2 - min
                                0.0);                                       // 3 - max

                            // Build distribution for delay parameters
                            NormalClippedDelay::ParamValues dDist(
                                Parameters::meanDelay[srcPop],              // 0 - mean
                                Parameters::delaySD[srcPop],                // 1 - sd
                                0.0,                                        // 2 - min
                                maxDelayMs[srcPop]);                        // 3 - max

                            // Create weight parameters
                            WeightUpdateModels::StaticPulseDendriticDelay::VarValues staticSynapseInit(
                                initVar<NormalClipped>(wDist),          // 0 - Wij (nA)
                                initVar<NormalClippedDelay>(dDist));    // 1 - delay (ms)

                            // Add synapse population
                            auto *synPop = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, GeNNModels::ExpCurr>(
                                synapseName, SYNAPSE_MATRIX_TYPE, NO_DELAY,
                                srcName, trgName,
                                {}, staticSynapseInit,
                                inhibitoryExpCurrParams, {});

                            // Set max connections
                            synPop->setMaxConnections(
                                GeNNUtils::calcFixedNumberTotalWithReplacementConnectorMaxConnections(numSrc, numTrg, numConnections));
                            synPop->setMaxDendriticDelayTimesteps(maxDendriticDelaySlots);
                        }

                    }
                }
            }
        }
    }

    std::cout << "Total neurons=" << totalNeurons << ", total synapses=" << totalSynapses << std::endl;

    // Finalise model
    model.finalize();
}
