#include "modelSpec.h"

#include "parameters.h"

//---------------------------------------------------------------------------
// Continuous
//---------------------------------------------------------------------------
class Continuous : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Continuous, 1, 1);

    SET_PARAM_NAMES({"vRest"});
    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(g) * ($(V_pre) - $(vRest)));\n");
};
IMPLEMENT_MODEL(Continuous);

void modelDefinition(NNmodel &model)
{
    model.setDT(1.0);
    model.setName("benchmark");
    model.setTiming(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitVarSnippet::Uniform::ParamValues vDist(
        -60.0,  // 0 - min
        -50.0); // 1 - max

    // LIF model parameters
    NeuronModels::LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        2.0);    // 6 - TauRefrac

    NeuronModels::LIF::ParamValues lifStimParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.11,    // 5 - Ioffset
        2.0);    // 6 - TauRefrac

    // LIF initial conditions
    // **TODO** uniform random
    NeuronModels::LIF::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),  // 0 - V
        0.0);    // 1 - RefracTime

    // Static synapse parameters
    Continuous::ParamValues continuousSynapseParams(
        -60.0);

    Continuous::VarValues continuousSynapseInit(
        0.00001);

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(Parameters::connectionProbability); // 0 - prob


    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::LIF>("Stim", Parameters::numPre,
                                                 lifStimParams, lifInit);
    model.addNeuronPopulation<NeuronModels::LIF>("Neurons", Parameters::numPost,
                                                 lifParams, lifInit);

    auto *syn = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>("Syn", SYNAPSE_MATRIX_TYPE, NO_DELAY,
                                                                                      "Stim", "Neurons",
                                                                                      continuousSynapseParams, continuousSynapseInit,
                                                                                      {}, {},
                                                                                      initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
}
