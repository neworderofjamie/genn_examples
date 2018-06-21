#include <cmath>
#include <vector>

#include "modelSpec.h"

// GeNN robotics includes
#include "genn_models/alpha_curr.h"

#include "parameters.h"

using namespace BoBRobotics;

//----------------------------------------------------------------------------
// STDPPower
//----------------------------------------------------------------------------
class STDPPower : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(STDPPower, 6, 1, 1, 1);

    SET_PARAM_NAMES({
      "tauPlus",    // 0 - Potentiation time constant (ms)
      "tauMinus",   // 1 - Depression time constant (ms)
      "lambda",     // 2 - Learning rate
      "alpha",      // 3 - Relative strength of depression and potentiation
      "mu",         // 4 - Power of potentiation weight update
      "weight0"});  // 5 - Reference weight


    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});

    SET_DERIVED_PARAMS({{"weight0Mu", [](const vector<double> &pars, double){ return std::pow(pars[5], 1.0 - pars[4]); }}});

    SET_PRE_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "$(preTrace) = ($(preTrace) * exp(-dt / $(tauPlus))) + 1.0;\n");
    SET_POST_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_post);\n"
        "$(postTrace) = ($(postTrace) * exp(-dt / $(tauMinus))) + 1.0;\n");

    SET_SIM_CODE(
        "$(addtoinSyn) = $(g);\n"
        "$(updatelinsyn);\n"
        "const scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0)\n"
        "{\n"
        "    const scalar timing = $(postTrace) * exp(-dt / $(tauMinus));\n"
        "    const scalar deltaG = -$(lambda) * $(alpha) * $(g) * timing;\n"
        "    $(g) += deltaG;\n"
        "}\n");
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
        {"ExpTC", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const vector<double> &pars, double){ return  pars[1] / pars[0]; }},
        {"PoissonExpMinusLambda", [](const vector<double> &pars, double dt){ return std::exp(-(pars[7] / 1000.0) * dt); }},
        {"IpoissonExpDecay", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[9]); }},
        {"IpoissonInit", [](const vector<double> &pars, double){ return pars[8] * (std::exp(1) / pars[9]); }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"Ipoisson", "scalar"}, {"Ipoisson2", "scalar"}});
};
IMPLEMENT_MODEL(LIFPoisson);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(0.1);
    model.setName("mad_2007");

    GENN_PREFERENCES::optimizeCode = true;
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::defaultSparseConnectivityMode = VarMode::LOC_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::autoChooseDevice = false;
    GENN_PREFERENCES::defaultDevice = 0;

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
        20.0,           // 0 - tauPlus
        20.0,           // 1 - tauMinus
        0.116,          // 2 - lambda
        1.057 * 0.1,    // 3 - alpha
        0.4,            // 4 - mu
        0.001);         // 5 - weight 0

    STDPPower::PreVarValues stdpPreInit(
        0.0);   // 0 - pre trace

    STDPPower::PostVarValues stdpPostInit(
        0.0);   // 1 - post trace

    // Synapse initial conditions
    WeightUpdateModels::StaticPulse::VarValues excitatorySynapseInit(
        Parameters::excitatoryPeakWeight);    // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues inhibitorySynapseInit(
        Parameters::excitatoryPeakWeight * Parameters::excitatoryInhibitoryRatio);    // 0 - Wij (nA)

    // Alpha current parameters
    GeNNModels::AlphaCurr::ParamValues alphaCurrParams(
        0.33);  // 0 - TauSyn (ms)
    GeNNModels::AlphaCurr::VarValues alphaCurrInit(
        0.0);   // 0 - x

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<LIFPoisson>("E", Parameters::numExcitatory, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<LIFPoisson>("I", Parameters::numInhibitory, lifParams, lifInit);

    auto *ee = model.addSynapsePopulation<STDPPower, GeNNModels::AlphaCurr>(
        "EE", SynapseMatrixType::RAGGED_INDIVIDUALG, Parameters::delayTimestep,
        "E", "E",
        stdpParams, excitatorySynapseInit, stdpPreInit, stdpPostInit,
        alphaCurrParams, alphaCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::AlphaCurr>(
        "EI", SynapseMatrixType::BITMASK_GLOBALG_INDIVIDUAL_PSM, Parameters::delayTimestep,
        "E", "I",
        {}, excitatorySynapseInit,
        alphaCurrParams, alphaCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::AlphaCurr>(
        "II", SynapseMatrixType::BITMASK_GLOBALG_INDIVIDUAL_PSM, Parameters::delayTimestep,
        "I", "I",
        {}, inhibitorySynapseInit,
        alphaCurrParams, alphaCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::AlphaCurr>(
        "IE", SynapseMatrixType::BITMASK_GLOBALG_INDIVIDUAL_PSM, Parameters::delayTimestep,
        "I", "E",
        {}, inhibitorySynapseInit,
        alphaCurrParams, alphaCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    i->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    // Configure plastic synaptic weights so that they can be downloaded to host
    ee->setWUVarMode("g", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    ee->setSparseConnectivityVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    model.finalize();
}
