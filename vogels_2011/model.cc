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
  model.setName("vogels_2011");

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
  const unsigned int MY_VOGELS_2011_ADDITIVE_STDP = weightUpdateModels.size();
  weightUpdateModels.push_back(weightUpdateModel());

  weightUpdateModels.back().pNames.reserve(5);
  weightUpdateModels.back().pNames =
  {
    "tau",      // 0 - Plasticity time constant (ms)
    "rho",      // 1 - Depression time constant (ms)
    "eta",      // 2 - Learning rate
    "Wmin",     // 3 - Minimum weight
    "Wmax",     // 4 - Maximum weight
  };

  weightUpdateModels.back().varNames.reserve(1);
  weightUpdateModels.back().varNames =
  {
    "g",        // 0 - conductance
  };

  weightUpdateModels.back().varTypes.reserve(1);
  weightUpdateModels.back().varTypes = {"scalar"};

  // Presynaptic spike update code
  weightUpdateModels.back().simCode =
    "$(addtoinSyn) = $(g);\n"
    "$(updatelinsyn);\n"
    "scalar dt = $(t) - $(sT_post); \n"
    "scalar timing = exp(-dt / $(tau)) - $(rho);\n"
    "scalar newWeight = $(g) - ($(eta) * timing);\n"
    "if(newWeight < $(Wmin))\n"
    "{\n"
    "  $(g) = $(Wmin);\n"
    "}\n"
    "else if(newWeight > $(Wmax))\n"
    "{\n"
    "  $(g) = $(Wmax);\n"
    "}\n"
    "else\n"
    "{\n"
    "  $(g) = newWeight;\n"
    "}\n";

  // code for post-synaptic spike
  weightUpdateModels.back().simLearnPost =
    "scalar dt = $(t) - $(sT_pre);\n"
    "scalar timing = exp(-dt / $(tau));\n"
    "scalar newWeight = $(g) - ($(eta) * timing);\n"
    "$(g) = (newWeight < $(Wmin)) ? $(Wmin) : newWeight;\n";

  // STDP rule requires pre and postsynaptic spike times
  weightUpdateModels.back().needPreSt = true;
  weightUpdateModels.back().needPostSt = true;

  //---------------------------------------------------------------------------
  // Build model
  //---------------------------------------------------------------------------
  // LIF model parameters
  double lifParams[7] =
  {
    0.2,    // 0 - C
    20.0,   // 1 - TauM
    -60.0,  // 2 - Vrest
    -60.0,  // 3 - Vreset
    -50.0,  // 4 - Vthresh
    0.2,    // 5 - Ioffset
    5.0,    // 6 - TauRefrac
  };

  // LIF initial conditions
  // **TODO** uniform random
  double lifInit[2] =
  {
    -55.0,  // 0 - V
    0.0,    // 1 - RefracTime
  };

  // Static synapse parameters
  double staticSynapseInit[1] =
  {
    0.03,    // 0 - Wij (nA)
  };

  // Additive STDP synapse parameters
  double vogels2011AdditiveSTDPParams[5] =
  {
    20.0,   // 0 - Tau
    0.12,   // 1 - rho
    0.05,  // 2 - eta
    -1.0,    // 3 - Wmin
    0.0,    // 4 - Wmax
  };

  double vogels2011AdditiveSTDPInit[1] =
  {
    0.0,  // 0 - g
  };

  // Exponential current parameters
  double excitatoryExpCurrParams[1] =
  {
    5.0,  // 0 - TauSyn (ms)
  };

  double inhibitoryExpCurrParams[1] =
  {
    10.0,  // 0 - TauSyn (ms)
  };

  // Create IF_curr neuron
  model.addNeuronPopulation("E", 2000, MY_LIF,
                            lifParams, lifInit);
  model.addNeuronPopulation("I", 500, MY_LIF,
                            lifParams, lifInit);

  model.addSynapsePopulation("EE", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, MY_EXP_CURR,
                             "E", "E",
                             staticSynapseInit, NULL,
                             NULL, excitatoryExpCurrParams);
  model.addSynapsePopulation("EI", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, MY_EXP_CURR,
                             "E", "I",
                             staticSynapseInit, NULL,
                             NULL, excitatoryExpCurrParams);
  model.addSynapsePopulation("II", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, MY_EXP_CURR,
                             "I", "I",
                             staticSynapseInit, NULL,
                             NULL, inhibitoryExpCurrParams);
  model.addSynapsePopulation("IE", MY_VOGELS_2011_ADDITIVE_STDP, SPARSE, INDIVIDUALG, NO_DELAY, MY_EXP_CURR,
                             "I", "E",
                             vogels2011AdditiveSTDPInit, vogels2011AdditiveSTDPParams,
                             NULL, inhibitoryExpCurrParams);
  model.finalize();
}