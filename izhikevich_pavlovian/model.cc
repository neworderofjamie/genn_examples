// GeNN includes
#include "modelSpec.h"

// GeNN robotics includes
#include "genn_models/stdp_dopamine.h"
#include "genn_utils/connectors.h"

// Model includes
#include "parameters.h"

using namespace BoBRobotics;

// Standard Izhikevich model with variable input current
class Izhikevich : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Izhikevich, 5, 3);

    SET_SIM_CODE(
        "scalar noise = ($(gennrand_uniform) * $(n) * 2.0) - $(n);\n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Iext)+noise)*DT; //at two times for numerical stability\n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Iext)+noise)*DT;\n"
        "$(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) >= 30.0");
    SET_RESET_CODE(
        "$(V)=$(c);\n"
        "$(U)+=$(d);\n");

    SET_PARAM_NAMES({"a", "b", "c", "d", "n"});
    SET_VARS({{"V","scalar"}, {"U", "scalar"}, {"Iext", "scalar"}});
};
IMPLEMENT_MODEL(Izhikevich);

void modelDefinition(NNmodel &model)
{
    // Use maths intrinsics rather than accurate trancendentals
    GENN_PREFERENCES::optimizeCode = true;
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;


    initGeNN();
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
    
    // Excitatory model parameters
    Izhikevich::ParamValues excParams(
        0.02,   // a
        0.2,    // b
        -65.0,  // c
        8.0,    // d
        6.5);   // n

    // Inhibitory model parameters
    Izhikevich::ParamValues inhParams(
        0.1,   // a
        0.2,    // b
        -65.0,  // c
        2.0,    // d
        6.5);   // n

    // LIF initial conditions
    Izhikevich::VarValues izkInit(
        -65.0,  // V
        -13.0,    // U
        0.0);   // Iext

    GeNNModels::STDPDopamine::ParamValues dopeParams(
        20.0,                       // 0 - Potentiation time constant (ms)
        20.0,                       // 1 - Depression time constant (ms)
        1000.0,                     // 2 - Synaptic tag time constant (ms)
        Parameters::tauD,           // 3 - Dopamine time constant (ms)
        0.1,                        // 4 - Rate of potentiation
        0.15,                       // 5 - Rate of depression
        0.0,                        // 6 - Minimum weight
        Parameters::maxExcWeight);  // 7 - Maximum weight

    GeNNModels::STDPDopamine::VarValues dopeInitVars(
        Parameters::initExcWeight,  // Synaptic weight
        0.0,                        // Synaptic tag
        0.0);                       // Time of last synaptic tag update


    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues inhSynInit(Parameters::inhWeight);

    // Create IF_curr neuron
    auto e = model.addNeuronPopulation<Izhikevich>("E", Parameters::numExcitatory, excParams, izkInit);
    auto i = model.addNeuronPopulation<Izhikevich>("I", Parameters::numInhibitory, inhParams, izkInit);

    auto ee = model.addSynapsePopulation<GeNNModels::STDPDopamine, PostsynapticModels::DeltaCurr>(
        "EE", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
        "E", "E",
        dopeParams, dopeInitVars,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    auto ei = model.addSynapsePopulation<GeNNModels::STDPDopamine, PostsynapticModels::DeltaCurr>(
        "EI", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
        "E", "I",
        dopeParams, dopeInitVars,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "II", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "I", "I",
        {}, inhSynInit,
        {}, {}, 
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "IE", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
        "I", "E",
        {}, inhSynInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    // Configure external input current variables so they can be uploaded from host
    e->setVarMode("Iext", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    i->setVarMode("Iext", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    // Configure spike variables so they can be downloaded to host
    e->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    i->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    // Configure synaptic weight variables so they can be downloaded to host
    ee->setWUVarMode("g", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    ei->setWUVarMode("g", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    model.finalize();
}