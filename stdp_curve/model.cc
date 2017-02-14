#include <cmath>
#include <vector>

#include "modelSpec.h"


//----------------------------------------------------------------------------
// ClosedFormLIF
//----------------------------------------------------------------------------
class ClosedFormLIF : public NeuronModels::Base
{
public:
    DECLARE_MODEL(ClosedFormLIF, 7, 2);

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
        {"Rmembrane", [](const vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_INIT_VALS({{"V", "scalar"}, {"RefracTime", "scalar"}});
};

IMPLEMENT_MODEL(ClosedFormLIF);

//----------------------------------------------------------------------------
// ExpCurr
//----------------------------------------------------------------------------
class ExpCurr : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(ExpCurr, 1, 0);

    SET_DECAY_CODE("$(inSyn)*=$(expDecay);");

    SET_CURRENT_CONVERTER_CODE("$(inSyn)");

    SET_PARAM_NAMES({"tau"});

    SET_DERIVED_PARAMS({{"expDecay", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});
};

IMPLEMENT_MODEL(ExpCurr);

//----------------------------------------------------------------------------
// STDPAdditive
//----------------------------------------------------------------------------
class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(STDPAdditive, 6, 1);

    SET_PARAM_NAMES({
      "tauPlus",  // 0 - Potentiation time constant (ms)
      "tauMinus", // 1 - Depression time constant (ms)
      "Aplus",    // 2 - Rate of potentiation
      "Aminus",   // 3 - Rate of depression
      "Wmin",     // 4 - Minimum weight
      "Wmax",     // 5 - Maximum weight
    });

    SET_INIT_VALS({{"g", "scalar"}});

    SET_SIM_CODE(
        "$(addtoinSyn) = $(g);\n"
        "$(updatelinsyn);\n"
        "scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar timing = exp(-dt / $(tauMinus));\n"
        "    scalar newWeight = $(g) - ($(Aminus) * timing);\n"
        "    $(g) = (newWeight < $(Wmin)) ? $(Wmin) : newWeight;\n"
        "}\n");
    SET_LEARN_POST_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar timing = exp(-dt / $(tauPlus));\n"
        "    scalar newWeight = $(g) + ($(Aplus) * timing);\n"
        "    $(g) = (newWeight > $(Wmax)) ? $(Wmax) : newWeight;\n"
        "}\n");
};

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(1.0);
  model.setName("stdp_curve");

  //---------------------------------------------------------------------------
  // Build model
  //---------------------------------------------------------------------------
  // LIF model parameters
  ClosedFormLIF::ParamValues lifParams(
      1.0,    // 0 - C
      20.0,   // 1 - TauM
      -70.0,  // 2 - Vrest
      -70.0,  // 3 - Vreset
      -51.0,  // 4 - Vthresh
      0.0,   // 5 - Ioffset
      2.0);  // 6 - TauRefrac

  // LIF initial conditions
  ClosedFormLIF::InitValues lifInit(
      -70.0,  // 0 - V
      0.0);    // 1 - RefracTime

  WeightUpdateModels::StaticPulse::InitValues staticSynapseInit(
      1.0);    // 0 - Wij (nA)

  // Additive STDP synapse parameters
  STDPAdditive::ParamValues additiveSTDPParams(
      16.7,   // 0 - TauPlus
      33.7,   // 1 - TauMinus
      0.005,  // 2 - APlus
      0.005,  // 3 - AMinus
      0.0,    // 4 - Wmin
      1.0);    // 5 - Wmax

  STDPAdditive::InitValues additiveSTDPInit(
      0.5);  // 0 - g

  // Exponential current parameters
  ExpCurr::ParamValues expCurrParams(
      5.0);  // 0 - TauSyn (ms)


  // Create IF_curr neuron
  model.addNeuronPopulation<NeuronModels::SpikeSource>("PreStim", 14, {}, {});
  model.addNeuronPopulation<NeuronModels::SpikeSource>("PostStim", 14, {}, {});
  model.addNeuronPopulation<ClosedFormLIF>("Excitatory", 14, lifParams, lifInit);

  model.addSynapsePopulation<STDPAdditive, PostsynapticModels::Izhikevich>(
          "PreStimToExcitatory", SPARSE, INDIVIDUALG, NO_DELAY,
          "PreStim", "Excitatory",
          additiveSTDPParams,  additiveSTDPInit,
          {}, {});
  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
          "PostStimToExcitatory", SPARSE, INDIVIDUALG, NO_DELAY,
          "PostStim", "Excitatory",
          {}, staticSynapseInit,
          expCurrParams, {});

  model.finalize();
}