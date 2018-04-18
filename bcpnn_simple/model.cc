// GeNN includes
#include "modelSpec.h"

// GeNN robotics includes
#include "exp_curr.h"
#include "lif.h"

// GeNN examples includes
#include "../common/bcpnn.h"

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(1.0);
    model.setName("bcpnn_simple");
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    LIF::ParamValues lifParams(
        0.25,   // 0 - C
        10.0,   // 1 - TauM
        -65.0,  // 2 - Vrest
        -70.0,  // 3 - Vreset
        -55.4,  // 4 - Vthresh
        0.0,   // 5 - Ioffset
        2.0);  // 6 - TauRefrac

    // LIF initial conditions
    LIF::VarValues lifInit(
        -65.0,  // 0 - V
        0.0);    // 1 - RefracTime

    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        2.0);    // 0 - Wij (nA)

    // BCPNN params
    BCPNNTwoTrace::ParamValues bcpnnParams(
        10.0,   // 0 - Time constant of presynaptic primary trace (ms)
        10.0,   // 1 - Time constant of postsynaptic primary trace (ms)
        1000.0, // 2 - Time constant of probability trace
        50.0,   // 3 - Maximum firing frequency (Hz)
        1.0,    // 4 - Maximum weight
        false,  // 5 - Should weights get applied to synapses
        true);  // 6 - Should weights be updated

    BCPNNTwoTrace::VarValues bcpnnInit(
        0.0,    // 0 - g
        0.0,    // 1 - PijStar
        0.0);   // 2 - lastUpdateTime

    BCPNNTwoTrace::PreVarValues bcpnnPreInit(
        0.0,    // 0 - ZiStar
        0.0);   // 1 - PiStar

    BCPNNTwoTrace::PostVarValues bcpnnPostInit(
        0.0,    // 0 - ZjStar
        0.0);   // 1 - PjStar

    // Exponential current parameters
    ExpCurr::ParamValues expCurrParams(
        2.5);  // 0 - TauSyn (ms)

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::SpikeSource>("PreStim", 1, {}, {});
    model.addNeuronPopulation<NeuronModels::SpikeSource>("PostStim", 1, {}, {});
    model.addNeuronPopulation<LIF>("Pre", 1, lifParams, lifInit);
    model.addNeuronPopulation<LIF>("Post", 1, lifParams, lifInit);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
            "PreStimToPre", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
            "PreStim", "Pre",
            {}, staticSynapseInit,
            expCurrParams, {});
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
            "PostStimToPost", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
            "PostStim", "Post",
            {}, staticSynapseInit,
            expCurrParams, {});

    model.addSynapsePopulation<BCPNNTwoTrace, ExpCurr>(
            "PreToPost", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
            "Pre", "Post",
            bcpnnParams,  bcpnnInit, bcpnnPreInit, bcpnnPostInit,
            expCurrParams, {});

    model.finalize();
}