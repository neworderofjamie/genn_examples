#include <cmath>
#include <vector>

#include "modelSpec.h"

// GeNN robotics includes
#include "genn_models/lif.h"
#include "genn_utils/connectors.h"

#include "parameters.h"

using namespace BoBRobotics;

//---------------------------------------------------------------------------
// Continuous
//---------------------------------------------------------------------------
class Continuous : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Continuous, 1, 1);

    SET_PARAM_NAMES({"vRest"});
    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "$(addtoinSyn) = $(g) * ($(V_pre) - $(vRest));\n"
        "$(updatelinsyn);\n");
};
IMPLEMENT_MODEL(Continuous);

void modelDefinition(NNmodel &model)
{
    // Enable new automatic initialisation mode
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;

    initGeNN();
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
    GeNNModels::LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        2.0);    // 6 - TauRefrac

    GeNNModels::LIF::ParamValues lifStimParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.11,    // 5 - Ioffset
        2.0);    // 6 - TauRefrac

    // LIF initial conditions
    // **TODO** uniform random
    GeNNModels::LIF::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),  // 0 - V
        0.0);    // 1 - RefracTime

    // Static synapse parameters
    Continuous::ParamValues continuousSynapseParams(
        -60.0);

    Continuous::VarValues continuousSynapseInit(
        0.00001);

    // Create IF_curr neuron
    model.addNeuronPopulation<GeNNModels::LIF>("Stim", Parameters::numPre,
                                               lifStimParams, lifInit);
    model.addNeuronPopulation<GeNNModels::LIF>("Neurons", Parameters::numPost,
                                               lifParams, lifInit);

    auto *syn = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>("Syn", SYNAPSE_MATRIX_TYPE, NO_DELAY,
                                                                                      "Stim", "Neurons",
                                                                                      continuousSynapseParams, continuousSynapseInit,
                                                                                      {}, {});
#if defined(SYNAPSE_MATRIX_CONNECTIVITY_SPARSE) || defined(SYNAPSE_MATRIX_CONNECTIVITY_RAGGED)
    syn->setMaxConnections(GeNNUtils::calcFixedProbabilityConnectorMaxConnections(Parameters::numPre, Parameters::numPost,
                                                                                  Parameters::connectionProbability));
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_SPARSE

    model.finalize();
}