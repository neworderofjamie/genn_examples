#include <cmath>
#include <vector>

#include "genn_models/lif.h"
#include "genn_models/exp_curr.h"

#include "modelSpec.h"

using namespace BoBRobotics;

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(1.0);
    model.setName("dendritic_delay");
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    GeNNModels::LIF::ParamValues lifParamVals(1.0, 20.0, -70.0, -70.0, -51.0, 0.0, 2.0);
    GeNNModels::LIF::VarValues lifInitVals(-70.0, 0.0);

    GeNNModels::ExpCurr::ParamValues expCurrParams(5.0);

    WeightUpdateModels::StaticPulseDendriticDelay::VarValues staticSynapseInitVals(10.0, uninitialisedVar());

    // Create IF_curr neuron
    auto *stim = model.addNeuronPopulation<NeuronModels::SpikeSource>("Stim", 1, {}, {});
    auto *excitatory = model.addNeuronPopulation<GeNNModels::LIF>("Excitatory", 10, lifParamVals, lifInitVals);

    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulseDendriticDelay, GeNNModels::ExpCurr>(
        "StimToExcitatory", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Stim", "Excitatory",
        {}, staticSynapseInitVals,
        expCurrParams, {});

    stim->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    excitatory->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    syn->setWUVarMode("d", VarMode::LOC_HOST_DEVICE_INIT_HOST);
    syn->setMaxDendriticDelaySlots(10);

    model.finalize();
}