// GeNN includes
#include "modelSpec.h"

// GeNN robotics includes
#include "../common/stdp_dopamine.h"

// Model includes
#include "parameters.h"

// Standard Izhikevich model with external input current
class Izhikevich : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Izhikevich, 0, 7);

    SET_SIM_CODE(
        "if ($(V) >= 30.0){\n"
        "   $(V)=$(c);\n"
        "   $(U)+=$(d);\n"
        "} \n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Iext))*DT; //at two times for numerical stability\n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Iext))*DT;\n"
        "$(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
        "if ($(V) > 30.0){   //keep this to not confuse users with unrealistiv voltage values \n"
        "  $(V)=30.0; \n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) >= 29.99");

    SET_VARS({{"V",     "scalar",   VarAccess::READ_WRITE},
              {"U",     "scalar",   VarAccess::READ_WRITE},
              {"Iext",  "scalar",   VarAccess::READ_ONLY},
              {"a",     "scalar",   VarAccess::READ_ONLY},
              {"b",     "scalar",   VarAccess::READ_ONLY},
              {"c",     "scalar",   VarAccess::READ_ONLY},
              {"d",     "scalar",   VarAccess::READ_ONLY}});
};
IMPLEMENT_MODEL(Izhikevich);

// Uniformly distributed input current
class UniformNoise : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(UniformNoise, 0, 1);

    SET_INJECTION_CODE("$(injectCurrent, ($(gennrand_uniform) * $(n) * 2.0) - $(n));\n");
    SET_VARS({{"n", "scalar", VarAccess::READ_ONLY}});
};
IMPLEMENT_MODEL(UniformNoise);

void modelDefinition(NNmodel &model)
{
    // Use maths intrinsics rather than accurate trancendentals
    GENN_PREFERENCES.optimizeCode = true;

    model.setDT(Parameters::timestepMs);
    model.setName("izhikevich_pavlovian");
    
#ifdef MEASURE_TIMING
    model.setTiming(true);
#endif
    
    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(
        Parameters::probabilityConnection); // 0 - prob

    // LIF initial conditions
    Izhikevich::VarValues excInit(
        -65.0,  // V
        -13.0,  // U
        0.0,    // Iext
        0.02,   // a
        0.2,    // b
        -65.0,  // c
        8.0);   // d

    // LIF initial conditions
    Izhikevich::VarValues inhInit(
        -65.0,  // V
        -13.0,  // U
        0.0,    // Iext
        0.1,    // a
        0.2,    // b
        -65.0,  // c
        2.0);   // d

    UniformNoise::VarValues currSourceInit(
        6.5);

    STDPDopamine::VarValues dopeInitVars(
        Parameters::initExcWeight,  // Synaptic weight
        0.0,                        // Synaptic tag
        0.0,                        // Time of last synaptic tag update
        20.0,                       // Potentiation time constant (ms)
        20.0,                       // Depression time constant (ms)
        1000.0,                     // Synaptic tag time constant (ms)
        Parameters::tauD,           // Dopamine time constant (ms)
        0.1,                        // Rate of potentiation
        0.15,                       // Rate of depression
        0.0,                        // Minimum weight
        Parameters::maxExcWeight);  // Maximum weight

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues inhSynInit(Parameters::inhWeight);

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<Izhikevich>("E", Parameters::numExcitatory, excInit);
    auto *i = model.addNeuronPopulation<Izhikevich>("I", Parameters::numInhibitory, inhInit);
    e->setVarImplementation("Iext", VarImplementation::INDIVIDUAL);
    i->setVarImplementation("Iext", VarImplementation::INDIVIDUAL);

    model.addCurrentSource<UniformNoise>("ECurr", "E", currSourceInit);
    model.addCurrentSource<UniformNoise>("ICurr", "I", currSourceInit);

    model.addSynapsePopulation<STDPDopamine, PostsynapticModels::DeltaCurr>(
        "EE", SynapseMatrixConnectivity::SPARSE, NO_DELAY,
        "E", "E",
        dopeInitVars, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<STDPDopamine, PostsynapticModels::DeltaCurr>(
        "EI", SynapseMatrixConnectivity::SPARSE, NO_DELAY,
        "E", "I",
        dopeInitVars, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "II", SynapseMatrixConnectivity::SPARSE, NO_DELAY,
        "I", "I",
        inhSynInit, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "IE", SynapseMatrixConnectivity::SPARSE, NO_DELAY,
        "I", "E",
        inhSynInit, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
}
