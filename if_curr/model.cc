#include "modelSpec.h"

// GeNN robotics includes
#include "exp_curr.h"
#include "lif.h"


void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(1.0);
  model.setName("if_curr");

  //---------------------------------------------------------------------------
  // Build model
  //---------------------------------------------------------------------------
  // LIF model parameters
  LIF::ParamValues lifParamVals(1.0, 20.0, -70.0, -70.0, -51.0, 0.0, 2.0);
  LIF::VarValues lifInitVals(-70.0, 0.0);

  ExpCurr::ParamValues expCurrParams(5.0);
  
  WeightUpdateModels::StaticPulse::VarValues staticSynapseInitVals(1.0);
  
  // Create IF_curr neuron
  model.addNeuronPopulation<NeuronModels::SpikeSource>("Stim", 1, {}, {});
  model.addNeuronPopulation<LIF>("Excitatory", 1, lifParamVals, lifInitVals);

  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>("StimToExcitatory", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY, 
                                                                       "Stim", "Excitatory",
                                                                       {}, staticSynapseInitVals,
                                                                       expCurrParams, {});

  model.finalize();
}