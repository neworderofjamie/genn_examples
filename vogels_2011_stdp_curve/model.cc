#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "../common/exp_curr.h"
#include "../common/lif.h"
#include "../common/vogels_2011.h"

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(1.0);
  model.setName("vogels_2011_stdp_curve");

  //---------------------------------------------------------------------------
  // Build model
  //---------------------------------------------------------------------------
  // LIF model parameters
  LIF::ParamValues lifParams(
      1.0,    // 0 - C
      20.0,   // 1 - TauM
      -70.0,  // 2 - Vrest
      -70.0,  // 3 - Vreset
      -51.0,  // 4 - Vthresh
      0.0,   // 5 - Ioffset
      2.0);  // 6 - TauRefrac

  // LIF initial conditions
  LIF::VarValues lifInit(
      -70.0,  // 0 - V
      0.0);    // 1 - RefracTime

  WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
      1.0);    // 0 - Wij (nA)

  // Additive STDP synapse parameters
  Vogels2011::ParamValues vogels2011Params(
      20.0,   // 0 - Tau
      0.12,   // 1 - rho
      0.005,  // 2 - eta
      -1.0,    // 3 - Wmin
      0.0);    // 4 - Wmax


  Vogels2011::VarValues vogels2011Init(
      -0.5);  // 0 - g

  // Exponential current parameters
  ExpCurr::ParamValues expCurrParams(
      5.0);  // 0 - TauSyn (ms)


  // Create IF_curr neuron
  model.addNeuronPopulation<NeuronModels::SpikeSource>("PreStim", 14, {}, {});
  model.addNeuronPopulation<NeuronModels::SpikeSource>("PostStim", 14, {}, {});
  model.addNeuronPopulation<LIF>("Excitatory", 14, lifParams, lifInit);

  model.addSynapsePopulation<Vogels2011, PostsynapticModels::DeltaCurr>(
          "PreStimToExcitatory", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
          "PreStim", "Excitatory",
          vogels2011Params,  vogels2011Init,
          {}, {});
  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
          "PostStimToExcitatory", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
          "PostStim", "Excitatory",
          {}, staticSynapseInit,
          expCurrParams, {});


  model.finalize();
}