// GeNN includes
#include "modelSpec.h"

// Common includes
#include "../common/stdp_dopamine.h"

// Model includes
#include "parameters.h"

// Standard Izhikevich model with variable input current
class Izhikevich : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Izhikevich, 4, 3);

    SET_SIM_CODE(
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Iext))*DT; //at two times for numerical stability\n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Iext))*DT;\n"
        "$(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) >= 30.0");
    SET_RESET_CODE(
        "$(V)=$(c);\n"
        "$(U)+=$(d);\n");

    SET_PARAM_NAMES({"a", "b", "c", "d"});
    SET_VARS({{"V","scalar"}, {"U", "scalar"}, {"Iext", "scalar"}});
};
IMPLEMENT_MODEL(Izhikevich);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(Parameters::timestepMs);
    model.setName("izhikevich_pavlovian");

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // Excitatory model parameters
    Izhikevich::ParamValues excParams(
        0.02,   // a
        0.2,    // b
        -65.0,  // c
        8.0);   // d

    // Inhibitory model parameters
    Izhikevich::ParamValues inhParams(
        0.1,   // a
        0.2,    // b
        -65.0,  // c
        2.0);   // d

    // LIF initial conditions
    Izhikevich::VarValues izkInit(
        -65.0,  // V
        -13.0,    // U
        0.0);   // Iext

    STDPDopamine::ParamValues dopeParams(
        20.0,               // 0 - Potentiation time constant (ms)
        20.0,               // 1 - Depression time constant (ms)
        1000.0,             // 2 - Synaptic tag time constant (ms)
        Parameters::tauD,   // 3 - Dopamine time constant (ms)
        0.1,                // 4 - Rate of potentiation
        0.15,               // 5 - Rate of depression
        0.0,                // 6 - Minimum weight
        4.0);               // 7 - Maximum weight

    STDPDopamine::VarValues dopeInitVars(
        1.0,    // Synaptic weight
        0.0,    // Synaptic tag
        0.0);   // Time of last synaptic tag update


    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues inhSynInit(-1.0);

    // Create IF_curr neuron
    model.addNeuronPopulation<Izhikevich>("E", Parameters::numExcitatory, excParams, izkInit);
    model.addNeuronPopulation<Izhikevich>("I", Parameters::numInhibitory, inhParams, izkInit);

    model.addSynapsePopulation<STDPDopamine, PostsynapticModels::DeltaCurr>(
        "EE", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "E", "E",
        dopeParams, dopeInitVars,
        {}, {});
    model.addSynapsePopulation<STDPDopamine, PostsynapticModels::DeltaCurr>(
        "EI", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "E", "I",
        dopeParams, dopeInitVars,
        {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "II", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "I", "I",
        {}, inhSynInit,
        {}, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "IE", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "I", "E",
        {}, inhSynInit,
        {}, {});

    // **TODO** set max connections
    model.finalize();
}