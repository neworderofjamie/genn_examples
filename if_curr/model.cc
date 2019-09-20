#include "modelSpec.h"

void modelDefinition(NNmodel &model)
{
  model.setDT(1.0);
  model.setName("if_curr");

  //---------------------------------------------------------------------------
  // Build model
  //---------------------------------------------------------------------------
  // LIF model parameters
  NeuronModels::LIF::ParamValues lifParamVals(1.0, 20.0, -70.0, -70.0, -51.0, 0.0, 2.0);
  NeuronModels::LIF::VarValues lifInitVals(-70.0, 0.0);

  PostsynapticModels::ExpCurr::ParamValues expCurrParams(5.0);
  
  WeightUpdateModels::StaticPulse::VarValues staticSynapseInitVals(1.0);
  
  // Create IF_curr neuron
  model.addNeuronPopulation<NeuronModels::SpikeSource>("Stim", 1, {}, {});
  model.addNeuronPopulation<NeuronModels::LIF>("Excitatory", 1, lifParamVals, lifInitVals);

  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
      "StimToExcitatory", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
      "Stim", "Excitatory",
      {}, staticSynapseInitVals,
      expCurrParams, {});
}
