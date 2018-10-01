#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "genn_models/exp_curr.h"
#include "genn_models/lif.h"

#include "../common/pfister_triplet.h"

#include "parameters.h"

using namespace BoBRobotics;

void modelDefinition(NNmodel &model)
{
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::defaultSparseConnectivityMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;

    initGeNN();
    model.setDT(1.0);
    model.setName("sjostrom_triplet");

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    GeNNModels::LIF::ParamValues lifParams(
        0.25,   // 0 - C
        10.0,   // 1 - TauM
        -65.0,  // 2 - Vrest
        -70.0,  // 3 - Vreset
        -55.4,  // 4 - Vthresh
        0.0,   // 5 - Ioffset
        2.0);  // 6 - TauRefrac

    // LIF initial conditions
    GeNNModels::LIF::VarValues lifInit(
        -65.0,  // 0 - V
        0.0);    // 1 - RefracTime

    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        2.0);    // 0 - Wij (nA)

    /*// Minimal triplet rule parameter set
    PfisterTriplet::ParamValues pfisterParams(
        16.8,                           // 0 - Tau plus
        33.7,                           // 1 - Tau minus
        101.0,                          // 2 - Tau X
        114.0,                          // 3 - Tau Y
        0.0,                            // 4 - A2+
        7.1E-3 * Parameters::aScale,    // 5 - A2-
        6.5E-3 * Parameters::aScale,    // 6 - A3+
        0.0,                            // 7 - A3-
        0.0,                            // 8 - Minimum weight
        1.0);                           // 9 - Maximum weight*/

    // Full triplet rule parameter set
    PfisterTriplet::ParamValues pfisterParams(
        16.8,                           // 0 - Tau plus
        33.7,                           // 1 - Tau minus
        101.0,                          // 2 - Tau X
        125.0,                          // 3 - Tau Y
        5.0E-10 * Parameters::aScale,   // 4 - A2+
        7.0E-3 * Parameters::aScale,    // 5 - A2-
        6.2E-3 * Parameters::aScale,    // 6 - A3+
        2.3E-4 * Parameters::aScale,    // 7 - A3-
        0.0,                            // 8 - Minimum weight
        1.0);                           // 9 - Maximum weight

    PfisterTriplet::VarValues pfisterInit(
        0.5);  // 0 - g

    PfisterTriplet::PreVarValues pfisterPreInit(
        0.0,    // 0 - r1
        0.0);   // 1 - r2

    PfisterTriplet::PostVarValues pfisterPostInit(
        0.0,    // 0 - o1
        0.0);   // 1 - o2

    // Exponential current parameters
    GeNNModels::ExpCurr::ParamValues expCurrParams(
        2.5);  // 0 - TauSyn (ms)

    std::cout << "Num neurons:" << Parameters::numNeurons << std::endl;

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::SpikeSource>("PreStim", Parameters::numNeurons, {}, {});
    model.addNeuronPopulation<NeuronModels::SpikeSource>("PostStim", Parameters::numNeurons, {}, {});
    model.addNeuronPopulation<GeNNModels::LIF>("Pre", Parameters::numNeurons, lifParams, lifInit);
    model.addNeuronPopulation<GeNNModels::LIF>("Post", Parameters::numNeurons, lifParams, lifInit);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::ExpCurr>(
            "PreStimToPre", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
            "PreStim", "Pre",
            {}, staticSynapseInit,
            expCurrParams, {},
            initConnectivity<InitSparseConnectivitySnippet::OneToOne>());

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::ExpCurr>(
            "PostStimToPost", SynapseMatrixType::RAGGED_GLOBALG, NO_DELAY,
            "PostStim", "Post",
            {}, staticSynapseInit,
            expCurrParams, {},
            initConnectivity<InitSparseConnectivitySnippet::OneToOne>());

    model.addSynapsePopulation<PfisterTriplet, GeNNModels::ExpCurr>(
            "PreToPost", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
            "Pre", "Post",
            pfisterParams, pfisterInit, pfisterPreInit, pfisterPostInit,
            expCurrParams, {},
            initConnectivity<InitSparseConnectivitySnippet::OneToOne>());

    model.finalize();
}