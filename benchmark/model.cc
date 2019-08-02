#include "modelSpec.h"

#include "parameters.h"

void modelDefinition(NNmodel &model)
{
    model.setDT(1.0);
    model.setName("benchmark");
    model.setTiming(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    NeuronModels::LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.5,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        -55.0,  // 0 - V
        0.0);    // 1 - RefracTime

    NeuronModels::PoissonNew::ParamValues poissonParams(
        Parameters::inputRate);  // 0 - rate [hz]

    NeuronModels::PoissonNew::VarValues poissonInit(
        0.0);   // 0 - time to spike [ms]

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        0.1);    // 0 - Wij (nA)

    // Exponential current parameters
    PostsynapticModels::ExpCurr::ParamValues expCurrParams(
        5.0);  // 0 - TauSyn (ms)

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(Parameters::connectionProbability); // 0 - prob

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::PoissonNew>("Poisson", Parameters::numNeurons,
                                                        poissonParams, poissonInit);
    model.addNeuronPopulation<NeuronModels::LIF>("Neurons", Parameters::numNeurons,
                                                 lifParams, lifInit);

    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "Syn", SYNAPSE_MATRIX_TYPE, NO_DELAY,
        "Poisson", "Neurons",
        {}, staticSynapseInit,
        expCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    syn->setSpanType(SynapseGroup::SpanType::POSTSYNAPTIC);
}
