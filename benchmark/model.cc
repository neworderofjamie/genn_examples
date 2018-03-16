#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "connectors.h"
#include "exp_curr.h"
#include "lif.h"

#include "parameters.h"

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
    InitVarSnippet::Normal::ParamValues gDist(
        0.0,    // 0 - mean
        0.1);   // 1 - sd

    // LIF model parameters
    LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    // **TODO** uniform random
    LIF::VarValues lifInit(
        -55.0,  // 0 - V
        0.0);    // 1 - RefracTime

    NeuronModels::PoissonNew::ParamValues poissonParams(
        10.0);      // 0 - firing rate

    NeuronModels::PoissonNew::VarValues poissonInit(
       0.0);     // 2 - SpikeTime

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        initVar<InitVarSnippet::Normal>(gDist));    // 0 - Wij (nA)
        //.0);

    // Exponential current parameters
    ExpCurr::ParamValues expCurrParams(
        5.0);  // 0 - TauSyn (ms)

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::PoissonNew>("Stim", Parameters::numPre,
                                poissonParams, poissonInit);
    model.addNeuronPopulation<LIF>("Neurons", Parameters::numPost,
                                   lifParams, lifInit);

    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>("Syn", SYNAPSE_MATRIX_TYPE, NO_DELAY,
                                                                                     "Stim", "Neurons",
                                                                                     {}, staticSynapseInit,
                                                                                     expCurrParams, {});
#ifdef SYNAPSE_MATRIX_CONNECTIVITY_SPARSE
    syn->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(Parameters::numPre, Parameters::numPost,
                                                                       Parameters::connectionProbability));
#endif  // SYNAPSE_MATRIX_CONNECTIVITY_SPARSE

    model.finalize();
}