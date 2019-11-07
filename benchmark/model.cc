#include "modelSpec.h"

#include "parameters.h"

// Create variable initialisation snippet to zero all weights aside from those
// that pass a fixed probability test.
class DenseFixedProbability : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(DenseFixedProbability, 2);

    SET_CODE(
        "const scalar r = $(gennrand_uniform);\n"
        "$(value) = (r < $(pconn)) ? $(gsyn) : 0.0;\n");
    SET_PARAM_NAMES({"pconn", "gsyn"});
};
IMPLEMENT_SNIPPET(DenseFixedProbability);

void modelDefinition(NNmodel &model)
{
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
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

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::PoissonNewAuto>("Poisson", Parameters::numNeurons, poissonInit);
    auto *n = model.addNeuronPopulation<NeuronModels::LIFAuto>("Neurons", Parameters::numNeurons, lifInit);

    // Configure spike variables so that they can be downloaded to host
    n->setSpikeLocation(VarLocation::HOST_DEVICE);

    // Exponential current parameters
    PostsynapticModels::ExpCurrAuto::VarValues expCurrInit(
        5.0);  // 0 - TauSyn (ms)

    // If connectivity is dense
    if(SYNAPSE_MATRIX_CONNECTIVITY == SynapseMatrixConnectivity::DENSE) {
        DenseFixedProbability::ParamValues fixedProb(Parameters::connectionProbability, 0.1);

        // Static synapse parameters
        WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
            initVar<DenseFixedProbability>(fixedProb));    // 0 - Wij (nA)

        auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurrAuto>(
            "Syn", SYNAPSE_MATRIX_CONNECTIVITY, NO_DELAY,
            "Poisson", "Neurons", staticSynapseInit, expCurrInit);
    }
    else {
        // Static synapse parameters
        WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
            0.1);    // 0 - Wij (nA)

        InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(Parameters::connectionProbability); // 0 - prob

        auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurrAuto>(
            "Syn", SYNAPSE_MATRIX_CONNECTIVITY, NO_DELAY,
            "Poisson", "Neurons", staticSynapseInit, expCurrInit,
            initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

        //yn->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);
        //syn->setNumThreadsPerSpike(8);
    }
}
