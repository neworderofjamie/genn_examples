// GeNN includes
#include "modelSpec.h"

// Common includes
#include "../common/connectors.h"
#include "../common/stdp_dopamine.h"

// Model includes
#include "parameters.h"

//---------------------------------------------------------------------------
// Standard Izhikevich model with variable input current
// and explicit threshold and reset
//---------------------------------------------------------------------------
class Izhikevich : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Izhikevich, 8, 3);

    SET_SIM_CODE(
        "$(V) += 0.5 * (($(k) * ($(V) - $(Vrest)) * ($(V) - $(Vthresh)))- $(U) + $(Isyn) + $(Iext)) * (DT / $(C)); //at two times for numerical stability\n"
        "$(V) += 0.5 * (($(k) * ($(V) - $(Vrest)) * ($(V) - $(Vthresh)))- $(U) + $(Isyn) + $(Iext)) * (DT / $(C));\n"
        "$(U) += $(a) * ($(b) * ($(V) - $(Vrest)) - $(U)) * DT;\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) >= $(Vthresh)");
    SET_RESET_CODE(
        "$(V) = $(c);\n"
        "$(U) += $(d);\n");

    SET_PARAM_NAMES({
        "C",        // 0
        "a",        // 1
        "b",        // 2
        "c",        // 3
        "d",        // 4
        "k",        // 5
        "Vrest",    // 6 - Resting membrane potential
        "Vthresh",  // 7 - Threshold potential
    });
    SET_VARS({
        {"V","scalar"},         // Membrane potential
        {"U", "scalar"},        // Recovery current
        {"Iext", "scalar"}});   // External input current
};
IMPLEMENT_MODEL(Izhikevich);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(Parameters::timestepMs);
    model.setName("ardin_webb_mb");

    //---------------------------------------------------------------------------
    // Neuron model parameters
    //---------------------------------------------------------------------------
    // PN model parameters
    Izhikevich::ParamValues pnParams(
        100.0,  // 0 - C
        0.3,    // 1 - a
        -0.2,   // 2 - b
        -65.0,  // 3 - c
        8.0,    // 4 - d
        2.0,    // 5 - k
        -60.0,  // 6 - Resting membrane potential
        -40.0); // 7 - Threshold

    // KC model parameters
    Izhikevich::ParamValues kcParams(
        4.0,    // 0 - C
        0.01,   // 1 - a
        -0.3,   // 2 - b
        -65.0,  // 3 - c
        8.0,    // 4 - d
        0.035,  // 5 - k
        -85.0,  // 6 - Resting membrane potential
        -25.0); // 7 - Threshold

    // EN model parameters
    Izhikevich::ParamValues enParams(
        100.0,  // 0 - C
        0.3,    // 1 - a
        -0.2,   // 2 - b
        -65.0,  // 3 - c
        8.0,    // 4 - d
        2.0,    // 5 - k
        -60.0,  // 6 - Resting membrane potential
        -40.0); // 7 - Threshold


    // Izhikevich initial conditions
    // **TODO** search for correct values
    Izhikevich::VarValues izkInit(
        -70.0,  // V
        -14.0,    // U
        0.0);   // Iext

    //---------------------------------------------------------------------------
    // Postsynaptic model parameters
    //---------------------------------------------------------------------------
    PostsynapticModels::ExpCond::ParamValues pnToKCPostsynapticParams(
        3.0,    // 0 - Synaptic time constant (ms)
        0.0);   // 1 - Reversal potential (mV)

    PostsynapticModels::ExpCond::ParamValues kcToENPostsynapticParams(
        8.0,    // 0 - Synaptic time constant (ms)
        0.0);   // 1 - Reversal potential (mV)

    //---------------------------------------------------------------------------
    // Weight update model parameters
    //---------------------------------------------------------------------------
    WeightUpdateModels::StaticPulse::VarValues pnToKCWeightUpdateParams(0.25 * 0.93 * Parameters::weightScale);

    STDPDopamine::ParamValues kcToENWeightUpdateParams(
        15.0,                                   // 0 - Potentiation time constant (ms)
        15.0,                                   // 1 - Depression time constant (ms)
        40.0,                                   // 2 - Synaptic tag time constant (ms)
        Parameters::tauD,                       // 3 - Dopamine time constant (ms)
        -1.0,                                   // 4 - Rate of potentiation
        1.0,                                    // 5 - Rate of depression
        0.0,                                    // 6 - Minimum weight
        2.0 * 8.0 * Parameters::weightScale);   // 7 - Maximum weight

    STDPDopamine::VarValues kcToENWeightUpdateInitVars(
        2.0 * 8.0 * Parameters::weightScale,    // Synaptic weight
        0.0,                                    // Synaptic tag
        0.0);                                   // Time of last synaptic tag update*/

    // Create neuron populations
    model.addNeuronPopulation<Izhikevich>("PN", Parameters::numPN, pnParams, izkInit);
    model.addNeuronPopulation<Izhikevich>("KC", Parameters::numKC, kcParams, izkInit);
    model.addNeuronPopulation<Izhikevich>("EN", Parameters::numEN, enParams, izkInit);

    auto pnToKC = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCond>(
        "pnToKC", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "PN", "KC",
        {}, pnToKCWeightUpdateParams,
        pnToKCPostsynapticParams, {});

    model.addSynapsePopulation<STDPDopamine, PostsynapticModels::ExpCond>(
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