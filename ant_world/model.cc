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
class LIFExtCurrent : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFExtCurrent, 9, 2);

    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0)\n"
        "{\n"
        "   scalar Iext = 0.0f;\n"
        "   if($(Iext) != NULL) {\n"
        "       unsigned int x = $(id) % (unsigned int)($(Width));\n"
        "       unsigned int y = $(id) / (unsigned int)($(Width));\n"
        "       const unsigned int index = (y * $(IextStep)) + x;\n"
        "       Iext = $(IextScale) * (scalar)$(Iext)[index];\n"
        "   }\n"
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
        "C",            // 0 -Membrane capacitance
        "TauM",         // 1 - Membrane time constant [ms]
        "Vrest",        // 2 - Resting membrane potential [mV]
        "Vreset",       // 3 - Reset voltage [mV]
        "Vthresh",      // 4 - Spiking threshold [mV]
        "Ioffset",      // 5 - Offset current
        "TauRefrac",    // 6 - Refractory time [ms]
        "IextScale",    // 7 - Scaling factor to apply to external current
        "Width",        // 8 - Width of population (pixels)
    });

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});

    SET_EXTRA_GLOBAL_PARAMS({{"Iext", "uint8_t *"}, {"IextStep", "unsigned int"}});
};
IMPLEMENT_MODEL(LIFExtCurrent);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(Parameters::timestepMs);
    model.setName("ant_world");

    //---------------------------------------------------------------------------
    // Neuron model parameters
    //---------------------------------------------------------------------------
    // LIF model parameters
    LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        2.0);   // 6 - TauRefrac

    // LIF model parameters
    LIFExtCurrent::ParamValues lifExtCurrParams(
        0.2,                            // 0 - C
        20.0,                           // 1 - TauM
        -60.0,                          // 2 - Vrest
        -60.0,                          // 3 - Vreset
        -50.0,                          // 4 - Vthresh
        0.0,                            // 5 - Ioffset
        2.0,                            // 6 - TauRefrac
        Parameters::inputCurrentScale,  // 7 - Scaling factor to apply to external current
        Parameters::inputWidth);        // 8 - Input width


    // LIF initial conditions
    LIF::VarValues lifInit(
        -60.0,  // 0 - V
        0.0);   // 1 - RefracTime

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
    model.addNeuronPopulation<LIFExtCurrent>("PN", Parameters::numPN, lifExtCurrParams, lifInit);
    model.addNeuronPopulation<LIF>("KC", Parameters::numKC, lifParams, lifInit);
    model.addNeuronPopulation<LIF>("EN", Parameters::numEN, lifParams, lifInit);

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