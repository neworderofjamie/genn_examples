#include <cmath>
#include <vector>

#include "modelSpec.h"

// GeNN robotics includes
#include "alpha_curr.h"
#include "connectors.h"
#include "lif.h"

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
        {"ExpTC", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const vector<double> &pars, double){ return  pars[1] / pars[0]; }},
        {"PoissonExpMinusLambda", [](const vector<double> &pars, double dt){ return std::exp(-(pars[7] / 1000.0) * dt); }},
        {"IpoissonExpDecay", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[9]); }},
        {"IpoissonInit", [](const vector<double> &pars, double dt){ return pars[8] * (1.0 - std::exp(-dt / pars[9])) * (pars[9] / dt); }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"Ipoisson", "scalar"}});
};
IMPLEMENT_MODEL(LIFPoisson);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(1.0);
    model.setName("mad_2007");

    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitVarSnippet::Normal::ParamValues vDist(
        5.7,    // 0 - mean
        7.2);   // 1 - standard deviation

    // LIF model parameters
    LIFPoisson::ParamValues lifParams(
        0.25,                               // 0 - C
        10.0,                               // 1 - TauM
        0.0,                                // 2 - Vrest
        0.0,                                // 3 - Vreset
        20.0,                               // 4 - Vthresh
        0.0,                                // 5 - Ioffset
        0.5,                                // 6 - TauRefrac
        (9000.0 * 2.32),                    // 7 - PoissonRate
        Parameters::excitatoryPeakWeight,   // 8 - PoissonWeight
        0.5);                               // 9 - IpoissonTau

    // LIF initial conditions
    LIFPoisson::VarValues lifInit(
        initVar<InitVarSnippet::Normal>(vDist), // 0 - V
        0.0,                                    // 1 - RefracTime
        0.0);                                   // 2 - Ipoisson

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues excitatoryStaticSynapseInit(
        Parameters::excitatoryPeakWeight);    // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues inhibitoryStaticSynapseInit(
        Parameters::excitatoryPeakWeight * Parameters::excitatoryInhibitoryRatio);    // 0 - Wij (nA)

    // Alpha current parameters
    AlphaCurr::ParamValues alphaCurrParams(
        0.33);  // 0 - TauSyn (ms)
    AlphaCurr::VarValues alphaCurrInit(
        0.0);   // 0 - x

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<LIFPoisson>("E", Parameters::numExcitatory, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<LIFPoisson>("I", Parameters::numInhibitory, lifParams, lifInit);

    auto *ee = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
        "EE", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        alphaCurrParams, alphaCurrInit);
    auto *ei = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
        "EI", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "E", "I",
        {}, excitatoryStaticSynapseInit,
        alphaCurrParams, alphaCurrInit);
    auto *ii = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
        "II", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "I", "I",
        {}, inhibitoryStaticSynapseInit,
        alphaCurrParams, alphaCurrInit);
    auto *ie = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, AlphaCurr>(
        "IE", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "I", "E",
        {}, inhibitoryStaticSynapseInit,
        alphaCurrParams, alphaCurrInit);

    ee->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(Parameters::numExcitatory, Parameters::numExcitatory,
                                                                      Parameters::probabilityConnection));
    ei->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(Parameters::numExcitatory, Parameters::numInhibitory,
                                                                      Parameters::probabilityConnection));
    ii->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(Parameters::numInhibitory, Parameters::numInhibitory,
                                                                      Parameters::probabilityConnection));
    ie->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(Parameters::numInhibitory, Parameters::numExcitatory,
                                                                      Parameters::probabilityConnection));

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    i->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    model.finalize();
}