#include <cmath>
#include <vector>

#include "modelSpec.h"

//----------------------------------------------------------------------------
// ClosedFormLIF
//----------------------------------------------------------------------------
class ClosedFormLIF : public NeuronModels::BaseSingleton<ClosedFormLIF>
{
public:
    DECLARE_PARAM_VALUES(7);
    DECLARE_INIT_VALUES(2);

    SET_SIM_CODE(
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
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_PARAM_NAMES({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "Ioffset",    // Offset current
        "TauRefrac"});

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const vector<double> &pars, double dt){ return  pars[1] / pars[0]; }}});

    SET_INIT_VALS({{"V", "scalar"}, {"RefracTime", "scalar"}});
};

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(1.0);
  model.setName("if_curr");

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
  auto lifParamVals = ClosedFormLIF::ParamValues(1.0, 20.0, -70.0, -70.0, -51.0, 0.0, 2.0);
  auto lifInitVals = ClosedFormLIF::InitValues(-70.0, 0.0);

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
  model.addNeuronPopulation<NeuronModels::SpikeSource>("Stim", 1, {}, {});
  model.addNeuronPopulation<ClosedFormLIF>("Excitatory", 1, lifParamVals, lifInitVals);

  model.addSynapsePopulation("StimToExcitatory", NSYNAPSE, ALLTOALL, INDIVIDUALG, NO_DELAY, MY_EXP_CURR,
                             "Stim", "Excitatory",
                             staticSynapseInit, NULL,
                             NULL, expCurrParams);

  model.finalize();
}