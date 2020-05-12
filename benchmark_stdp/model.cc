#include "modelSpec.h"

#include "../common/stdp_additive.h"

#include "parameters.h"

void modelDefinition(NNmodel &model)
{
    model.setDT(1.0);
    model.setName("benchmark_stdp");
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

    NeuronModels::PoissonNew::ParamValues poissonInputParams(
        Parameters::inputRate);  // 0 - rate [hz]
        
    NeuronModels::PoissonNew::ParamValues poissonOutputParams(
        Parameters::outputRate);  // 0 - rate [hz]
    
    NeuronModels::PoissonNew::VarValues poissonInit(
        0.0);   // 0 - time to spike [ms]

    // STDP parameters
    STDPAdditive::ParamValues stdpParams(
        20.0,   // 0 - Potentiation time constant (ms)
        20.0,   // 1 - Depression time constant (ms)
        0.001,  // 2 - Rate of potentiation
        0.001,  // 3 - Rate of depression
        0.0,    // 4 - Minimum weight
        1.0);   // 5 - Maximum weight

    STDPAdditive::VarValues stdpInit(
        0.5);    // 0 - Wij (nA)

    // Exponential current parameters
    PostsynapticModels::ExpCurr::ParamValues expCurrParams(
        5.0);  // 0 - TauSyn (ms)

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(Parameters::connectionProbability); // 0 - prob

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::PoissonNew>("Pre", Parameters::numNeurons,
                                                        poissonInputParams, poissonInit);
    model.addNeuronPopulation<NeuronModels::PoissonNew>("Post", Parameters::numNeurons,
                                                        poissonOutputParams, poissonInit);

    auto *syn = model.addSynapsePopulation<STDPAdditive, PostsynapticModels::ExpCurr>(
        "Syn", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "Pre", "Post",
        stdpParams, stdpInit,
        expCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    syn->setSpanType(SynapseGroup::SpanType::POSTSYNAPTIC);
}
