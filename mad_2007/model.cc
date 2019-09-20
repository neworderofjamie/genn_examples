#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "../common/alpha_curr.h"

#include "parameters.h"


//----------------------------------------------------------------------------
// STDPPower
//----------------------------------------------------------------------------
class STDPPower : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(STDPPower, 7, 1, 1, 1);

    SET_PARAM_NAMES({
        "tauPlus",      // 0 - Potentiation time constant (ms)
        "tauMinus",     // 1 - Depression time constant (ms)
        "lambda",       // 2 - Learning rate
        "alpha",        // 3 - Relative strength of depression and potentiation
        "mu",           // 4 - Power of potentiation weight update
        "weight0",      // 5 - Reference weight (nA)
        "denDelay"});   // 6 - Dendritic delay (ms)


    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});

    SET_DERIVED_PARAMS({
        {"weight0Mu", [](const std::vector<double> &pars, double){ return std::pow(pars[5], 1.0 - pars[4]); }},
        {"denDelayStep", [](const std::vector<double> &pars, double dt){ return std::floor(pars[6] / dt) - 1.0; }}
    });

    SET_PRE_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "$(preTrace) = ($(preTrace) * exp(-dt / $(tauPlus))) + 1.0;\n");
    SET_POST_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_post);\n"
        "$(postTrace) = ($(postTrace) * exp(-dt / $(tauMinus))) + 1.0;\n");

    SET_SIM_CODE(
        "const scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0)\n"
        "{\n"
        "    const scalar timing = $(postTrace) * exp(-dt / $(tauMinus));\n"
        "    const scalar deltaG = -$(lambda) * $(alpha) * $(g) * timing;\n"
        "    $(g) += deltaG;\n"
        "}\n"
        "$(addToInSynDelay, $(g), (unsigned int)$(denDelayStep));\n");
    SET_LEARN_POST_CODE(
        "const scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0)\n"
        "{\n"
        "    const scalar timing = $(preTrace) * exp(-dt / $(tauPlus));\n"
        "    const scalar deltaG = $(lambda) * $(weight0Mu) * pow($(g), $(mu)) * timing;\n"
        "    $(g) += deltaG;\n"
        "}\n");

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};
IMPLEMENT_MODEL(STDPPower);

//----------------------------------------------------------------------------
// LIFPoisson
//----------------------------------------------------------------------------
//! Leaky integrate-and-fire neuron solved algebraically with direct, alpha-shaped Poisson input
class LIFPoisson : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFPoisson, 10, 4);

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
        "  scalar alpha = (($(Isyn) + $(Ioffset) + $(Ipoisson2)) * $(Rmembrane)) + $(Vrest);\n"
        "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else\n"
        "{\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
        "$(Ipoisson2) = (DT * $(IpoissonExpDecay) * $(Ipoisson)) + ($(IpoissonExpDecay) * $(Ipoisson2));\n"
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
        {"IpoissonInit", [](const std::vector<double> &pars, double){ return pars[8] * (std::exp(1) / pars[9]); }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"Ipoisson", "scalar"}, {"Ipoisson2", "scalar"}});
};
IMPLEMENT_MODEL(LIFPoisson);

void modelDefinition(NNmodel &model)
{
    GENN_PREFERENCES.optimizeCode = true;
    GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MANUAL;
    GENN_PREFERENCES.manualDeviceID = 0;

    model.setDT(0.1);
    model.setTimePrecision(TimePrecision::DOUBLE);
    model.setName("mad_2007");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTiming(Parameters::measureTiming);


    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitVarSnippet::Normal::ParamValues vDist(
        5.7,    // 0 - mean
        7.2);   // 1 - standard deviation

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(
        Parameters::probabilityConnection); // 0 - prob

    // LIF model parameters
    LIFPoisson::ParamValues lifParams(
        0.25,                               // 0 - C
        10.0,                               // 1 - TauM
        0.0,                                // 2 - Vrest
        0.0,                                // 3 - Vreset
        20.0,                               // 4 - Vthresh
        0.0,                                // 5 - Ioffset
        0.5,                                // 6 - TauRefrac
        Parameters::externalInputRate,      // 7 - PoissonRate
        Parameters::excitatoryPeakWeight,   // 8 - PoissonWeight
        0.33);                               // 9 - IpoissonTau

    // LIF initial conditions
    LIFPoisson::VarValues lifInit(
        initVar<InitVarSnippet::Normal>(vDist), // 0 - V
        0.0,                                    // 1 - RefracTime
        0.0,                                    // 2 - Ipoisson
        0.0);                                   // 3 - Ipoisson2

    // STDP parameters
    STDPPower::ParamValues stdpParams(
        20.0,                   // 0 - tauPlus
        20.0,                   // 1 - tauMinus
        0.1,                    // 2 - lambda
        1.057 * 0.1,            // 3 - alpha
        0.4,                    // 4 - mu
        0.001,                  // 5 - weight 0
        Parameters::delayMs);   // 6 - dendritic delay

    STDPPower::PreVarValues stdpPreInit(
        0.0);   // 0 - pre trace

    STDPPower::PostVarValues stdpPostInit(
        0.0);   // 1 - post trace

    // Synapse initial conditions
    WeightUpdateModels::StaticPulse::VarValues excitatorySynapseInit(
        Parameters::excitatoryPeakWeight);    // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues inhibitorySynapseInit(
        Parameters::excitatoryPeakWeight * Parameters::excitatoryInhibitoryRatio);    // 0 - Wij (nA)

#ifdef STATIC
    InitVarSnippet::Normal::ParamValues wDist(
        0.04565,    // 0 - mean
        0.00399);   // 1 - standard deviation

    WeightUpdateModels::StaticPulse::VarValues eeSynapseInit(
        initVar<InitVarSnippet::Normal>(wDist));    // 0 - Wij (nA)
#endif
    // Alpha current parameters
    AlphaCurr::ParamValues alphaCurrParams(
        0.33);  // 0 - TauSyn (ms)
    AlphaCurr::VarValues alphaCurrInit(
        0.0);   // 0 - x

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<LIFPoisson>("E", Parameters::numExcitatory, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<LIFPoisson>("I", Parameters::numInhibitory, lifParams, lifInit);

#ifdef STATIC
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
        "EE", SynapseMatrixType::SPARSE_INDIVIDUALG, Parameters::delayTimestep - 1,
        "E", "E",
        {}, eeSynapseInit,
        alphaCurrParams, alphaCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
#else
    // **NOTE** in order for the weights to remain stable it is important 
    // that delay is dendritic with matching back propagation delay
    auto *ee = model.addSynapsePopulation<STDPPower, AlphaCurr>(
        "EE", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "E", "E",
        stdpParams, excitatorySynapseInit, stdpPreInit, stdpPostInit,
        alphaCurrParams, alphaCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    ee->setMaxDendriticDelayTimesteps(Parameters::delayTimestep);
    ee->setBackPropDelaySteps(Parameters::delayTimestep - 4);
#endif
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
        "EI", SynapseMatrixType::SPARSE_GLOBALG_INDIVIDUAL_PSM, Parameters::delayTimestep - 1,
        "E", "I",
        {}, excitatorySynapseInit,
        alphaCurrParams, alphaCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
        "II", SynapseMatrixType::BITMASK_GLOBALG_INDIVIDUAL_PSM, Parameters::delayTimestep - 1,
        "I", "I",
        {}, inhibitorySynapseInit,
        alphaCurrParams, alphaCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
        "IE", SynapseMatrixType::BITMASK_GLOBALG_INDIVIDUAL_PSM, Parameters::delayTimestep - 1,
        "I", "E",
        {}, inhibitorySynapseInit,
        alphaCurrParams, alphaCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeLocation(VarLocation::HOST_DEVICE);
    i->setSpikeLocation(VarLocation::HOST_DEVICE);

    // Configure plastic synaptic weights so that they can be downloaded to host
#ifndef STATIC
    ee->setWUVarLocation("g", VarLocation::HOST_DEVICE);
    ee->setSparseConnectivityLocation(VarLocation::HOST_DEVICE);
#endif
}
