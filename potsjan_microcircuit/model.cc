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
    DECLARE_MODEL(LIFPoisson, 0, 13);

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

    SET_VARS({{"V",             "scalar",   VarAccess::READ_WRITE},
              {"RefracTime",    "scalar",   VarAccess::READ_WRITE},
              {"Ipoisson",      "scalar",   VarAccess::READ_WRITE},
              {"C",             "scalar",   VarAccess::READ_ONLY},      // Membrane capacitance
              {"TauM",          "scalar",   VarAccess::READ_ONLY},      // Membrane time constant [ms]
              {"Vrest",         "scalar",   VarAccess::READ_ONLY},      // Resting membrane potential [mV]
              {"Vreset",        "scalar",   VarAccess::READ_ONLY},      // Reset voltage [mV]
              {"Vthresh",       "scalar",   VarAccess::READ_ONLY},      // Spiking threshold [mV]
              {"Ioffset",       "scalar",   VarAccess::READ_ONLY},      // Offset current
              {"TauRefrac",     "scalar",   VarAccess::READ_ONLY},      // Refractory time [ms]
              {"PoissonRate",   "scalar",   VarAccess::READ_ONLY},      // Poisson input rate [Hz]
              {"PoissonWeight", "scalar",   VarAccess::READ_ONLY},      // How much current each poisson spike adds [nA]
              {"IpoissonTau",   "scalar",   VarAccess::READ_ONLY}});    // Time constant of poisson spike integration [ms]

    SET_DERIVED_PARAMS_NAMED({
        {"ExpTC", [](const std::map<std::string, double> &vars, double dt){ return std::exp(-dt / vars.at("TauM")); }},
        {"Rmembrane", [](const std::map<std::string, double> &vars, double){ return vars.at("TauM") / vars.at("C"); }},
        {"PoissonExpMinusLambda", [](const std::map<std::string, double> &vars, double dt){ return std::exp(-(vars.at("PoissonRate") / 1000.0) * dt); }},
        {"IpoissonExpDecay", [](const std::map<std::string, double> &vars, double dt){ return std::exp(-dt / vars.at("IpoissonTau")); }},
        {"IpoissonInit", [](const std::map<std::string, double> &vars, double dt){ return vars.at("PoissonWeight") * (1.0 - std::exp(-dt / vars.at("IpoissonTau"))) * (vars.at("IpoissonTau") / dt); }}});
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
    SET_ROW_BUILD_STATE_VARS({{"x", "scalar", 0.0}, {"c", "unsigned int", 0}});

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

    SET_VARS({{"g", "scalar", VarAccess::READ_ONLY},
              {"d", "uint8_t", VarAccess::READ_ONLY}});

    SET_SIM_CODE("$(addToInSynDelay, $(g), $(d));\n");
    //SET_SIM_CODE("$(addToInSynDelayVec, $(g.x), $(d.x), $(g.y), $(d.y));\n");
};
IMPLEMENT_MODEL(StaticPulseDendriticDelayHalf);

void modelDefinition(NNmodel &model)
{
    model.setDT(Parameters::dtMs);
    model.setName("potjans_microcircuit");
    model.setTiming(Parameters::measureTiming);
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    //model.setDefaultNarrowSparseIndEnabled(true);
    model.setMergePostsynapticModels(true);

    GENN_PREFERENCES.optimizeCode = true;
    GENN_PREFERENCES.generateLineInfo = true;

    InitVarSnippet::Normal::ParamValues vDist(
        -58.0, // 0 - mean
        5.0);  // 1 - sd

    // Exponential current parameters
    PostsynapticModels::ExpCurrAuto::VarValues excitatoryExpCurrInit(
        0.5);  // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurrAuto::VarValues inhibitoryExpCurrInit(
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
            LIFPoisson::VarValues lifInit(
                initVar<InitVarSnippet::Normal>(vDist), // 0 - V
                0.0,                                    // 1 - RefracTime
                0.0,                                    // 2 - Ipoisson
                0.25,                                   // 0 - C
                10.0,                                   // 1 - TauM
                -65.0,                                  // 2 - Vrest
                -65.0,                                  // 3 - Vreset
                -50.0,                                  // 4 - Vthresh
                extInputCurrent,                        // 5 - Ioffset
                2.0,                                    // 6 - TauRefrac
                extInputRate,                           // 7 - PoissonRate
                extWeight,                              // 8 - PoissonWeight
                0.5);                                   // 9 - IpoissonTau

            // Create population
            const unsigned int popSize = Parameters::getScaledNumNeurons(layer, pop);
            auto *neuronPop = model.addNeuronPopulation<LIFPoisson>(popName, popSize, lifInit);

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
                            auto *synPop = model.addSynapsePopulation<StaticPulseDendriticDelayHalf, PostsynapticModels::ExpCurrAuto>(
                                synapseName, SynapseMatrixConnectivity::PROCEDURAL, NO_DELAY,
                                srcName, trgName,
                                staticSynapseInit, excitatoryExpCurrInit,
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
                            auto *synPop = model.addSynapsePopulation<StaticPulseDendriticDelayHalf, PostsynapticModels::ExpCurrAuto>(
                                synapseName, SynapseMatrixConnectivity::PROCEDURAL, NO_DELAY,
                                srcName, trgName,
                                staticSynapseInit, inhibitoryExpCurrInit,
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
