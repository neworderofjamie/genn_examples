#include <cmath>
#include <vector>

#include "modelSpec.h"

class ClosedFormLIF : public dpclass
{
public:
    virtual double calculateDerivedParameter(int index, vector<double> pars, double dt = 1.0)
    {
        switch (index)
        {
        case 0: // ExpTC
            return std::exp(-dt / pars[1]);
        case 1: // Rmembrane
            return pars[1] / pars[0];

        }
        return -1;
    }
};


void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(1.0);
  model.setName("stdp_curve");

  //---------------------------------------------------------------------------
  // Create LIF neuron
  //---------------------------------------------------------------------------
  const unsigned int MY_LIF = nModels.size();
  nModels.push_back(neuronModel());

  nModels.back().varNames.reserve(2);
  nModels.back().varNames = {"V", "RefracTime"};

  nModels.back().varTypes.reserve(2);
  nModels.back().varTypes = {"scalar", "scalar"};

  nModels.back().pNames.reserve(7);
  nModels.back().pNames =
  {
    "C",          // Membrane capacitance
    "TauM",       // Membrane time constant [ms]
    "Vrest",      // Resting membrane potential [mV]
    "Vreset",     // Reset voltage [mV]
    "Vthresh",    // Spiking threshold [mV]
    "Ioffset",    // Offset current
    "TauRefrac",  // Refractory time [ms]
  };

  nModels.back().dpNames.reserve(2);
  nModels.back().dpNames = {"ExpTC", "Rmembrane"};
  nModels.back().dps = new ClosedFormLIF();

  //TODO: replace the resetting in the following with BRIAN-like threshold and resetting
  nModels.back().simCode =
    "if ($(RefracTime) <= 0.0)\n"
    "{\n"
    "  if ($(V) >= $(Vthresh))\n"
    "  {\n"
    "    $(V) = $(Vreset);\n"
    "    $(RefracTime) = $(TauRefrac);\n"
    "  }\n"
    "  scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);\n"
    "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
    "}\n"
    "else\n"
    "{\n"
    "  $(RefracTime) -= DT;\n"
    "}\n";

  nModels.back().thresholdConditionCode = "$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)";


  //---------------------------------------------------------------------------
  // Create exponential current postsynaptic mechanism
  //---------------------------------------------------------------------------
  const unsigned int MY_EXP_CURR = postSynModels.size();
  postSynModels.push_back(postSynModel());

  postSynModels.back().pNames.reserve(1);
  postSynModels.back().pNames = {"tau"};

  postSynModels.back().dpNames.reserve(1);
  postSynModels.back().dpNames = {"expDecay"};
  postSynModels.back().dps = new expDecayDp;

  postSynModels.back().postSynDecay= "$(inSyn)*=$(expDecay);\n";
  postSynModels.back().postSyntoCurrent= "$(inSyn)";

  //---------------------------------------------------------------------------
  // Create STDP rule
  //---------------------------------------------------------------------------
  const unsigned int MY_ADDITIVE_STDP = weightUpdateModels.size();
  weightUpdateModels.push_back(weightUpdateModel());

  weightUpdateModels.back().pNames.reserve(4);
  weightUpdateModels.back().pNames =
  {
    "tauPlus",  // 0 - Potentiation time constant (ms)
    "tauMinus", // 1 - Depression time constant (ms)
    "Aplus",    // 2 - Rate of potentiation
    "Aminus"    // 3 - Rate of depression
  };

  weightUpdateModels.back().varNames.reserve(2);
  weightUpdateModels.back().varNames =
  {
    "g",        // 0 - conductance
    "gLearnt",  // 1 - learnt conductance
  };

  weightUpdateModels.back().varTypes.reserve(2);
  weightUpdateModels.back().varTypes = {"scalar", "scalar"};

  // Presynaptic spike update code
  weightUpdateModels.back().simCode =
    "$(addtoinSyn) = $(g);\n"
    "$(updatelinsyn);\n"
    "scalar dt = $(t) - $(sT_post); \n"
    "if (dt > 0)\n"
    "{\n"
    "  scalar timing = exp(-dt / $(tauMinus));\n"
    "  $(gLearnt) -= ($(Aminus) * timing);\n"
    "}\n";

  // code for post-synaptic spike
  weightUpdateModels.back().simLearnPost =
    "scalar dt = $(t) - $(sT_pre);\n"
    "if (dt > 0)\n"
    "{\n"
    "  scalar timing = exp(-dt / $(tauPlus));\n"
    "  $(gLearnt) += ($(Aplus) * timing);\n"
    "}\n";

  // STDP rule requires pre and postsynaptic spike times
  weightUpdateModels.back().needPreSt = true;
  weightUpdateModels.back().needPostSt = true;

  //---------------------------------------------------------------------------
  // Build model
  //---------------------------------------------------------------------------
  // LIF model parameters
  double lifParams[7] =
  {
    1.0,    // 0 - C
    20.0,   // 1 - TauM
    -70.0,  // 2 - Vrest
    -70.0,  // 3 - Vreset
    -51.0,  // 4 - Vthresh
    0.0,   // 5 - Ioffset
    2.0,    // 6 - TauRefrac
  };

  // LIF initial conditions
  double lifInit[2] =
  {
    -70.0,  // 0 - V
    0.0,    // 1 - RefracTime
  };

  // Static synapse parameters
  double staticSynapseInit[1] =
  {
    1.0,    // 0 - Wij (nA)
  };

  // Additive STDP synapse parameters
  double additiveSTDPParams[4] =
  {
    16.7,   // 0 - TauPlus
    33.7,   // 1 - TauMinus
    0.005,  // 2 - APlus
    0.005,  // 3 - AMinus
  };

  double additiveSTDPInit[2] =
  {
    1.0,  // 0 - g
    0.5,  // 1 - GLearnt
  };

  // Exponential current parameters
  double expCurrParams[1] =
  {
    5.0,  // 0 - TauSyn (ms)
  };

  // Create IF_curr neuron
  model.addNeuronPopulation("PreStim", 14, SPIKESOURCE, {}, {});
  model.addNeuronPopulation("PostStim", 14, SPIKESOURCE, {}, {});
  model.addNeuronPopulation("Excitatory", 14, MY_LIF,
                            lifParams, lifInit);

  model.addSynapsePopulation("PreStimToExcitatory", MY_ADDITIVE_STDP, SPARSE, INDIVIDUALG, NO_DELAY, MY_EXP_CURR,
                             "PreStim", "Excitatory",
                             additiveSTDPInit, additiveSTDPParams,
                             NULL, expCurrParams);
  model.addSynapsePopulation("PostStimToExcitatory", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, MY_EXP_CURR,
                             "PostStim", "Excitatory",
                             staticSynapseInit, NULL,
                             NULL, expCurrParams);

  model.finalize();
}