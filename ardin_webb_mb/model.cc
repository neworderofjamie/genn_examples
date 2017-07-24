// GeNN includes
#include "modelSpec.h"

// Common includes
#include "../common/connectors.h"
#include "../common/exp_curr.h"
#include "../common/lif.h"
#include "../common/stdp_dopamine.h"

// Model includes
#include "parameters.h"

//---------------------------------------------------------------------------
// Standard LIF model extended to take an additional
// input current from an extra global variable
//---------------------------------------------------------------------------
class LIFVarOffset : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFVarOffset, 6, 3);

    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0)\n"
        "{\n"
        "   scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);\n"
        "   $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else\n"
        "{\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "TauRefrac"});

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"Ioffset", "scalar"}});
};
IMPLEMENT_MODEL(LIFVarOffset);

//---------------------------------------------------------------------------
// Standard LIF model extended to take an additional
// input current from an extra global variable
//---------------------------------------------------------------------------
class LIFExtCurrent : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFExtCurrent, 6, 3);

    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0)\n"
        "{\n"
        "   const scalar Iext = *($(Iext) + $(IextOffset) + $(id));\n"
        "   scalar alpha = (($(Isyn) + $(Ioffset) + Iext) * $(Rmembrane)) + $(Vrest);\n"
        "   $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else\n"
        "{\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "TauRefrac"});

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"Ioffset", "scalar"}});

    SET_EXTRA_GLOBAL_PARAMS({{"Iext", "scalar *"}, {"IextOffset", "unsigned int"}});
};
IMPLEMENT_MODEL(LIFExtCurrent);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(Parameters::timestepMs);
    model.setName("ardin_webb_mb");

    //---------------------------------------------------------------------------
    // Neuron model parameters
    //---------------------------------------------------------------------------
    // LIF model parameters
    LIFExtCurrent::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        2.0);    // 5 - TauRefrac

    // LIF initial conditions
    LIFExtCurrent::VarValues lifInit(
        -60.0,  // 0 - V
        0.0,    // 1 - RefracTime
        0.0);   // 2 - Ioffset

    //---------------------------------------------------------------------------
    // Postsynaptic model parameters
    //---------------------------------------------------------------------------
    ExpCurr::ParamValues pnToKCPostsynapticParams(
        3.0);   // 0 - Synaptic time constant (ms)

    ExpCurr::ParamValues kcToENPostsynapticParams(
        8.0);   // 0 - Synaptic time constant (ms)

    //---------------------------------------------------------------------------
    // Weight update model parameters
    //---------------------------------------------------------------------------
    WeightUpdateModels::StaticPulse::VarValues pnToKCWeightUpdateParams(Parameters::pnToKCWeight);

    STDPDopamine::ParamValues kcToENWeightUpdateParams(
        15.0,                       // 0 - Potentiation time constant (ms)
        15.0,                       // 1 - Depression time constant (ms)
        40.0,                       // 2 - Synaptic tag time constant (ms)
        Parameters::tauD,           // 3 - Dopamine time constant (ms)
        -1.0,                       // 4 - Rate of potentiation
        1.0,                        // 5 - Rate of depression
        0.0,                        // 6 - Minimum weight
        Parameters::kcToENWeight);  // 7 - Maximum weight

    STDPDopamine::VarValues kcToENWeightUpdateInitVars(
        Parameters::kcToENWeight,   // Synaptic weight
        0.0,                        // Synaptic tag
        0.0);                       // Time of last synaptic tag update

    // Create neuron populations
    model.addNeuronPopulation<LIFExtCurrent>("PN", Parameters::numPN, lifParams, lifInit);
    model.addNeuronPopulation<LIFVarOffset>("KC", Parameters::numKC, lifParams, lifInit);
    model.addNeuronPopulation<LIFVarOffset>("EN", Parameters::numEN, lifParams, lifInit);

    auto pnToKC = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "pnToKC", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "PN", "KC",
        {}, pnToKCWeightUpdateParams,
        pnToKCPostsynapticParams, {});

    model.addSynapsePopulation<STDPDopamine, ExpCurr>(
        "kcToEN", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "KC", "EN",
        kcToENWeightUpdateParams, kcToENWeightUpdateInitVars,
        kcToENPostsynapticParams, {});


    // Calculate max connections
    const unsigned int maxConn = calcFixedNumberPreConnectorMaxConnections(Parameters::numPN, Parameters::numKC,
                                                                           Parameters::numPNSynapsesPerKC);

    std::cout << "Max connections:" << maxConn << std::endl;
    pnToKC->setMaxConnections(maxConn);

    model.finalize();
}