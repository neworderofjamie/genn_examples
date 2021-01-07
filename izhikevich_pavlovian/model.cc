// GeNN includes
#include "modelSpec.h"

// GeNN robotics includes
#include "../common/stdp_dopamine.h"

// Model includes
#include "parameters.h"

// Standard Izhikevich model
class IzhikevichDopamine : public NeuronModels::Base
{
public:
    DECLARE_MODEL(IzhikevichDopamine, 6, 3);

    SET_SIM_CODE(
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT; //at two times for numerical stability\n"
        "$(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn))*DT;\n"
        "$(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
        "const unsigned int timestep = (unsigned int)($(t) / DT);\n"
        "const bool injectDopamine = (($(dTime)[timestep / 32] & (1 << (timestep % 32))) != 0);\n"
        "if(injectDopamine) {\n"
        "   const scalar dopamineDT = $(t) - $(prev_seT);\n"
        "   const scalar dopamineDecay = exp(-dopamineDT / $(tauD));\n"
        "   $(D) = ($(D) * dopamineDecay) + $(dStrength);\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(V) >= 30.0");
    SET_RESET_CODE(
        "$(V)=$(c);\n"
        "$(U)+=$(d);\n");

    SET_PARAM_NAMES({"a", "b", "c", "d", "tauD", "dStrength"});
    SET_VARS({{"V","scalar"}, {"U", "scalar"}, {"D", "scalar"}});
    
    SET_EXTRA_GLOBAL_PARAMS({{"dTime", "uint32_t*"}});
};
IMPLEMENT_MODEL(IzhikevichDopamine);

// Uniformly distributed input current
class StimAndNoiseSource : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(StimAndNoiseSource, 2, 2);

    SET_INJECTION_CODE(
        "scalar current = ($(gennrand_uniform) * $(n) * 2.0) - $(n);\n"
        "if($(startStim) != $(endStim) && $(t) >= $(stimTimes)[$(startStim)]) {\n"
        "   current += $(stimMagnitude);\n"
        "   $(startStim)++;\n"
        "}\n"
        "$(injectCurrent, current);\n");
    
    SET_PARAM_NAMES({"n", "stimMagnitude"});
    SET_VARS( {{"startStim", "unsigned int"}, {"endStim", "unsigned int", VarAccess::READ_ONLY}} );
    SET_EXTRA_GLOBAL_PARAMS( {{"stimTimes", "scalar*"}} );
};
IMPLEMENT_MODEL(StimAndNoiseSource);

void modelDefinition(NNmodel &model)
{
    // Use maths intrinsics rather than accurate trancendentals
    GENN_PREFERENCES.optimizeCode = true;

    model.setDT(Parameters::timestepMs);
    model.setName("izhikevich_pavlovian");
    
    model.setTiming(Parameters::measureTiming);
    
    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(
        Parameters::probabilityConnection); // 0 - prob
    
    // Excitatory model parameters
    IzhikevichDopamine::ParamValues excParams(
        0.02,                           // a
        0.2,                            // b
        -65.0,                          // c
        8.0,                            // d
        Parameters::tauD,               // Dopamine time constant [ms]
        Parameters::dopamineStrength);  // Dopamine strength
    
    // Excitatory initial conditions
    IzhikevichDopamine::VarValues excInit(
        -65.0,  // V
        -13.0,  // U
        0.0);   // D

    // Inhibitory model parameters
    NeuronModels::Izhikevich::ParamValues inhParams(
        0.1,    // a
        0.2,    // b
        -65.0,  // c
        2.0);   // d

    // Inhibitory initial conditions
    NeuronModels::Izhikevich::VarValues inhInit(
        -65.0,  // V
        -13.0); // U
        
    StimAndNoiseSource::ParamValues currSourceParams(
        6.5,                            // n
        Parameters::stimuliCurrent);    // Stimuli magnitude
    
    StimAndNoiseSource::VarValues currSourceInit(
        uninitialisedVar(),   // startStim
        uninitialisedVar());  // endStim
    
    STDPDopamine::ParamValues dopeParams(
        20.0,                       // 0 - Potentiation time constant (ms)
        20.0,                       // 1 - Depression time constant (ms)
        1000.0,                     // 2 - Synaptic tag time constant (ms)
        Parameters::tauD,           // 3 - Dopamine time constant (ms)
        0.1,                        // 4 - Rate of potentiation
        0.15,                       // 5 - Rate of depression
        0.0,                        // 6 - Minimum weight
        Parameters::maxExcWeight);  // 7 - Maximum weight

    STDPDopamine::VarValues dopeInitVars(
        Parameters::initExcWeight,  // Synaptic weight
        0.0);                       // Synaptic tag

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues inhSynInit(Parameters::inhWeight);

    // Create IF_curr neuron and enable spike recording
    auto *e = model.addNeuronPopulation<IzhikevichDopamine>("E", Parameters::numExcitatory, excParams, excInit);
    auto *i = model.addNeuronPopulation<NeuronModels::Izhikevich>("I", Parameters::numInhibitory, inhParams, inhInit);
    e->setSpikeRecordingEnabled(true);
    i->setSpikeRecordingEnabled(true);

    model.addCurrentSource<StimAndNoiseSource>("ECurr", "E", currSourceParams, currSourceInit);
    model.addCurrentSource<StimAndNoiseSource>("ICurr", "I", currSourceParams, currSourceInit);

    model.addSynapsePopulation<STDPDopamine, PostsynapticModels::DeltaCurr>(
        "EE", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "E", "E",
        dopeParams, dopeInitVars,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<STDPDopamine, PostsynapticModels::DeltaCurr>(
        "EI", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "E", "I",
        dopeParams, dopeInitVars,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "II", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "I", "I",
        {}, inhSynInit,
        {}, {}, 
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "IE", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "I", "E",
        {}, inhSynInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
}
