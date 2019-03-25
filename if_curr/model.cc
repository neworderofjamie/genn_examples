#include "modelSpec.h"

// GeNN robotics includes
#include "genn_models/exp_curr.h"
#include "genn_models/lif.h"

using namespace BoBRobotics;

void modelDefinition(NNmodel &model)
{
  model.setDT(1.0);
  model.setName("if_curr");

  //---------------------------------------------------------------------------
  // Build model
  //---------------------------------------------------------------------------
  // LIF model parameters
  GeNNModels::LIF::ParamValues lifParamVals(1.0, 20.0, -70.0, -70.0, -51.0, 0.0, 2.0);
  GeNNModels::LIF::VarValues lifInitVals(-70.0, 0.0);

  GeNNModels::ExpCurr::ParamValues expCurrParams(5.0);
  
  WeightUpdateModels::StaticPulse::VarValues staticSynapseInitVals(1.0);
  
  // Create IF_curr neuron
  model.addNeuronPopulation<NeuronModels::SpikeSource>("Stim", 1, {}, {});
  model.addNeuronPopulation<GeNNModels::LIF>("Excitatory", 1, lifParamVals, lifInitVals);

  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::ExpCurr>(
      "StimToExcitatory", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
      "Stim", "Excitatory",
      {}, staticSynapseInitVals,
      expCurrParams, {});
}
