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
    NeuronModels::LIFAuto::VarValues lifInit(
        -55.0,  // 0 - V
        0.0,    // 1 - RefracTime
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.5,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    NeuronModels::PoissonNewAuto::VarValues poissonInit(
        0.0,                    // 0 - time to spike [ms]
        Parameters::inputRate); // 1 - rate [hz]

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        0.1);    // 0 - Wij (nA)

    // Exponential current parameters
    PostsynapticModels::ExpCurrAuto::VarValues expCurrInit(
        5.0);  // 0 - TauSyn (ms)

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(Parameters::connectionProbability); // 0 - prob

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::PoissonNewAuto>("Poisson", Parameters::numNeurons, poissonInit);
    model.addNeuronPopulation<NeuronModels::LIFAuto>("Neurons", Parameters::numNeurons, lifInit);

    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurrAuto>(
        "Syn", SYNAPSE_MATRIX_CONNECTIVITY, NO_DELAY,
        "Poisson", "Neurons", staticSynapseInit, expCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    //yn->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);
    //syn->setNumThreadsPerSpike(8);
}
