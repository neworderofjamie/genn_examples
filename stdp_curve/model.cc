#include "modelSpec.h"

#include "../common/stdp_additive.h"

void modelDefinition(NNmodel &model)
{
    model.setDT(1.0);
    model.setName("stdp_curve");

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    NeuronModels::LIF::ParamValues lifParams(
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        -70.0,  // 2 - Vrest
        -70.0,  // 3 - Vreset
        -51.0,  // 4 - Vthresh
        0.0,   // 5 - Ioffset
        2.0);  // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        -70.0,  // 0 - V
        0.0);    // 1 - RefracTime

    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        8.0);    // 0 - Wij (nA)

    // Additive STDP synapse parameters
    STDPAdditive::ParamValues additiveSTDPParams(
        16.7,   // 0 - TauPlus
        33.7,   // 1 - TauMinus
        0.005,  // 2 - APlus
        0.005,  // 3 - AMinus
        0.0,    // 4 - Wmin
        1.0);    // 5 - Wmax

    STDPAdditive::VarValues additiveSTDPInit(
        0.5);  // 0 - g

    // Exponential current parameters
    PostsynapticModels::ExpCurr::ParamValues expCurrParams(
        5.0);  // 0 - TauSyn (ms)

    // Unitialised spike source array
    NeuronModels::SpikeSourceArray::VarValues spikeSourceInit(
        uninitialisedVar(),     // 0 - startSpike
        uninitialisedVar());    // 1 - endSpike

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::SpikeSourceArray>("PreStim", 14, {}, spikeSourceInit);
    model.addNeuronPopulation<NeuronModels::SpikeSourceArray>("PostStim", 14, {}, spikeSourceInit);
    model.addNeuronPopulation<NeuronModels::LIF>("Excitatory", 14, lifParams, lifInit);

    model.addSynapsePopulation<STDPAdditive, PostsynapticModels::DeltaCurr>(
            "PreStimToExcitatory", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
            "PreStim", "Excitatory",
            additiveSTDPParams,  additiveSTDPInit,
            {}, {},
            initConnectivity<InitSparseConnectivitySnippet::OneToOne>());
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
            "PostStimToExcitatory", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
            "PostStim", "Excitatory",
            {}, staticSynapseInit,
            expCurrParams, {},
            initConnectivity<InitSparseConnectivitySnippet::OneToOne>());
}
