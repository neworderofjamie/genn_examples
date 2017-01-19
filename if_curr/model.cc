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
  model.setName("if_curr");

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

  // Exponential current parameters
  double expCurrParams[1] =
  {
    5.0,  // 0 - TauSyn (ms)
  };

  // Create IF_curr neuron
  model.addNeuronPopulation("Stim", 1, SPIKESOURCE, {}, {});
  model.addNeuronPopulation("Excitatory", 1, MY_LIF,
                            lifParams, lifInit);

  model.addSynapsePopulation("StimToExcitatory", NSYNAPSE, ALLTOALL, INDIVIDUALG, NO_DELAY, MY_EXP_CURR,
                             "Stim", "Excitatory",
                             staticSynapseInit, NULL,
                             NULL, expCurrParams);

  model.finalize();
}