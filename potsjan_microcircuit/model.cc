// GeNN includes
#include "modelSpec.h"

// Genn examples includes
#include "../common/normal_distribution.h"

// Model includes
#include "parameters.h"

//----------------------------------------------------------------------------
// LIFPoisson
//----------------------------------------------------------------------------
//! Leaky integrate-and-fire neuron solved algebraically
class LIFPoisson : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFPoisson, 10, 3);

    SET_SIM_CODE(
        "scalar p = 1.0f;\n"
        "unsigned int numPoissonSpikes = 0;\n"
        "do\n"
        "{\n"
        "    numPoissonSpikes++;\n"
        "    p *= $(gennrand_uniform);\n"
        "} while (p > $(PoissonExpMinusLambda));\n"
        "$(Ipoisson) += $(IpoissonInit) * (scalar)(numPoissonSpikes - 1);\n"
        "if ($(RefracTime) <= 0.0)\n"
        "{\n"
        "  scalar alpha = (($(Isyn) + $(Ioffset) + $(Ipoisson)) * $(Rmembrane)) + $(Vrest);\n"
        "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else\n"
        "{\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
        "$(Ipoisson) *= $(IpoissonExpDecay);\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "C",                // Membrane capacitance
        "TauM",             // Membrane time constant [ms]
        "Vrest",            // Resting membrane potential [mV]
        "Vreset",           // Reset voltage [mV]
        "Vthresh",          // Spiking threshold [mV]
        "Ioffset",          // Offset current
        "TauRefrac",        // Refractory time [ms]
        "PoissonRate",      // Poisson input rate [Hz]
        "PoissonWeight",    // How much current each poisson spike adds [nA]
        "IpoissonTau"});     // Time constant of poisson spike integration [ms]


    SET_DERIVED_PARAMS({
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }},
        {"PoissonExpMinusLambda", [](const std::vector<double> &pars, double dt){ return std::exp(-(pars[7] / 1000.0) * dt); }},
        {"IpoissonExpDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[9]); }},
        {"IpoissonInit", [](const std::vector<double> &pars, double dt){ return pars[8] * (1.0 - std::exp(-dt / pars[9])) * (pars[9] / dt); }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"Ipoisson", "scalar"}});
};
IMPLEMENT_MODEL(LIFPoisson);

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

//----------------------------------------------------------------------------
// FixedNumberTotalWithReplacement
//----------------------------------------------------------------------------
//! Initialises variable by sampling from the uniform distribution
class FixedNumberTotalWithReplacement : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(FixedNumberTotalWithReplacement, 1);

    SET_ROW_BUILD_CODE(
        "const unsigned int rowLength = $(preCalcRowLength)[$(id)];\n"
        "const scalar u = $(gennrand_uniform);\n"
        "x += (1.0 - x) * (1.0 - pow(u, 1.0 / (scalar)(rowLength - c)));\n"
        "unsigned int postIdx = (unsigned int)(x * $(num_post));\n"
        "postIdx = (postIdx < $(num_post)) ? postIdx : ($(num_post) - 1);\n"
        "$(addSynapse, postIdx + $(id_post_begin));\n"
        "c++;\n"
        "if(c >= rowLength) {\n"
        "   $(endRow);\n"
        "}\n");
    SET_ROW_BUILD_STATE_VARS({{"x", "scalar", 0.0},{"c", "unsigned int", 0}});

    SET_PARAM_NAMES({"total"});
    SET_EXTRA_GLOBAL_PARAMS({{"preCalcRowLength", "unsigned int*"}})

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected, and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, pars[0], (double)numPost / ((double)numPre * (double)numPost));
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected, and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, pars[0], (double)numPre / ((double)numPre * (double)numPost));
        });
};
IMPLEMENT_SNIPPET(FixedNumberTotalWithReplacement);

class StaticPulseDendriticDelayHalf : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(StaticPulseDendriticDelayHalf, 0, 2);

    SET_VARS({{"g", "scalar"},{"d", "uint8_t"}});

    SET_SIM_CODE("$(addToInSynDelay, $(g), $(d));\n");
};
IMPLEMENT_MODEL(StaticPulseDendriticDelayHalf);

void modelDefinition(NNmodel &model)
{
    model.setDT(Parameters::dtMs);
    model.setName("potjans_microcircuit");
    model.setTiming(Parameters::measureTiming);
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setMergePostsynapticModels(true);

    GENN_PREFERENCES.optimizeCode = true;

    InitVarSnippet::Normal::ParamValues vDist(
        -58.0, // 0 - mean
        5.0);  // 1 - sd

    // LIF initial conditions
    LIFPoisson::VarValues lifInit(
        initVar<InitVarSnippet::Normal>(vDist), // 0 - V
        0.0,                                    // 1 - RefracTime
        0.0);                                   // 2 - Ipoisson

    // Exponential current parameters
    PostsynapticModels::ExpCurr::ParamValues excitatoryExpCurrParams(
        0.5);  // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurr::ParamValues inhibitoryExpCurrParams(
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
            LIFPoisson::ParamValues lifParams(
                0.25,               // 0 - C
                10.0,               // 1 - TauM
                -65.0,              // 2 - Vrest
                -65.0,              // 3 - Vreset
                -50.0,              // 4 - Vthresh
                extInputCurrent,    // 5 - Ioffset
                2.0,                // 6 - TauRefrac
                extInputRate,       // 7 - PoissonRate
                extWeight,          // 8 - PoissonWeight
                0.5);               // 9 - IpoissonTau

            // Create population
            const unsigned int popSize = Parameters::getScaledNumNeurons(layer, pop);
            auto *neuronPop = model.addNeuronPopulation<LIFPoisson>(popName, popSize,
                                                                    lifParams, lifInit);

            // Make recordable on host
#ifdef USE_ZERO_COPY
            neuronPop->setSpikeLocation(VarLocation::ZERO_COPY);
#else
            neuronPop->setSpikeLocation(VarLocation::HOST_DEVICE);
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

            // Loop through source populations and layers
            for(unsigned int srcLayer = 0; srcLayer < Parameters::LayerMax; srcLayer++) {
                for(unsigned int srcPop = 0; srcPop < Parameters::PopulationMax; srcPop++) {
                    const std::string srcName = Parameters::getPopulationName(srcLayer, srcPop);

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
                        const double prob = (double)numConnections / ((double)Parameters::getScaledNumNeurons(srcLayer, srcPop) * (double)Parameters::getScaledNumNeurons(trgLayer, trgPop));
                        std::cout << "\tConnection between '" << srcName << "' and '" << trgName << "': numConnections=" << numConnections << "(" << prob << "), meanWeight=" << meanWeight << ", weightSD=" << weightSD << ", meanDelay=" << Parameters::meanDelay[srcPop] << ", delaySD=" << Parameters::delaySD[srcPop] << std::endl;

                        // Build parameters for fixed number total connector
                        FixedNumberTotalWithReplacement::ParamValues connectParams(
                            numConnections);                            // 0 - number of connections

                        totalSynapses += numConnections;

                        // Build unique synapse name
                        const std::string synapseName = srcName + "_" + trgName;

                        // Determine matrix type
                        const SynapseMatrixType matrixType = Parameters::proceduralConnectivity
                            ? SynapseMatrixType::PROCEDURAL_PROCEDURALG
                            : SynapseMatrixType::SPARSE_INDIVIDUALG;

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
                            StaticPulseDendriticDelayHalf::VarValues staticSynapseInit(
                                initVar<NormalClipped>(wDist),          // 0 - Wij (nA)
                                initVar<NormalClippedDelay>(dDist));    // 1 - delay (ms)

                            // Add synapse population
                            auto *synPop = model.addSynapsePopulation<StaticPulseDendriticDelayHalf, PostsynapticModels::ExpCurr>(
                                synapseName, matrixType, NO_DELAY, srcName, trgName,
                                {}, staticSynapseInit,
                                excitatoryExpCurrParams, {},
                                initConnectivity<FixedNumberTotalWithReplacement>(connectParams));

                            // Set max dendritic delay and span type
                            synPop->setMaxDendriticDelayTimesteps(maxDendriticDelaySlots);
                            synPop->setSpanType(Parameters::presynapticParallelism ? SynapseGroup::SpanType::PRESYNAPTIC : SynapseGroup::SpanType::POSTSYNAPTIC);
                            if(Parameters::presynapticParallelism) {
                                synPop->setNumThreadsPerSpike(Parameters::numThreadsPerSpike);
                            }
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
                            StaticPulseDendriticDelayHalf::VarValues staticSynapseInit(
                                initVar<NormalClipped>(wDist),          // 0 - Wij (nA)
                                initVar<NormalClippedDelay>(dDist));    // 1 - delay (ms)

                            // Add synapse population
                            auto *synPop = model.addSynapsePopulation<StaticPulseDendriticDelayHalf, PostsynapticModels::ExpCurr>(
                                synapseName, matrixType, NO_DELAY, srcName, trgName,
                                {}, staticSynapseInit,
                                inhibitoryExpCurrParams, {},
                                initConnectivity<FixedNumberTotalWithReplacement>(connectParams));

                            // Set max dendritic delay and span type
                            synPop->setMaxDendriticDelayTimesteps(maxDendriticDelaySlots);
                            synPop->setSpanType(Parameters::presynapticParallelism ? SynapseGroup::SpanType::PRESYNAPTIC : SynapseGroup::SpanType::POSTSYNAPTIC);
                            if(Parameters::presynapticParallelism) {
                                synPop->setNumThreadsPerSpike(Parameters::numThreadsPerSpike);
                            }
                        }

                    }
                }
            }
        }
    }

    std::cout << "Total neurons=" << totalNeurons << ", total synapses=" << totalSynapses << std::endl;
}
