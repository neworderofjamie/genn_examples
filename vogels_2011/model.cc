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

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});
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
// Vogels2011
//----------------------------------------------------------------------------
class Vogels2011 : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Vogels2011, 5, 1);

    SET_PARAM_NAMES({
        "tau",      // 0 - Plasticity time constant (ms)
        "rho",      // 1 - Target rate
        "eta",      // 2 - Learning rate
        "Wmin",     // 3 - Minimum weight
        "Wmax",     // 4 - Maximum weight
    });

    SET_VARS({{"g", "scalar"}});

    SET_SIM_CODE(
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
        "}\n");

    SET_LEARN_POST_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "scalar timing = exp(-dt / $(tau));\n"
        "scalar newWeight = $(g) - ($(eta) * timing);\n"
        "$(g) = (newWeight < $(Wmin)) ? $(Wmin) : newWeight;\n");

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};

IMPLEMENT_MODEL(Vogels2011);


void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(1.0);
  model.setName("vogels_2011");


  //---------------------------------------------------------------------------
  // Build model
  //---------------------------------------------------------------------------
  // LIF model parameters
  ClosedFormLIF::ParamValues lifParams(
      0.2,    // 0 - C
      20.0,   // 1 - TauM
      -60.0,  // 2 - Vrest
      -60.0,  // 3 - Vreset
      -50.0,  // 4 - Vthresh
      0.2,    // 5 - Ioffset
      5.0);    // 6 - TauRefrac

  // LIF initial conditions
  // **TODO** uniform random
  ClosedFormLIF::VarValues lifInit(
      -55.0,  // 0 - V
      0.0);    // 1 - RefracTime

  // Static synapse parameters
  WeightUpdateModels::StaticPulse::VarValues excitatoryStaticSynapseInit(
      0.03);    // 0 - Wij (nA)

  WeightUpdateModels::StaticPulse::VarValues inhibitoryStaticSynapseInit(
      -0.03);    // 0 - Wij (nA)

  // Additive STDP synapse parameters
  Vogels2011::ParamValues vogels2011AdditiveSTDPParams(
      20.0,   // 0 - Tau
      0.12,   // 1 - rho
      0.005,  // 2 - eta
      -1.0,    // 3 - Wmin
      0.0);    // 4 - Wmax

  Vogels2011::VarValues vogels2011AdditiveSTDPInit(
      0.0);  // 0 - g

  // Exponential current parameters
  ExpCurr::ParamValues excitatoryExpCurrParams(
      5.0);  // 0 - TauSyn (ms)

  ExpCurr::ParamValues inhibitoryExpCurrParams(
      10.0);  // 0 - TauSyn (ms)

  // Create IF_curr neuron
  model.addNeuronPopulation<ClosedFormLIF>("E", 2000,
                            lifParams, lifInit);
  model.addNeuronPopulation<ClosedFormLIF>("I", 500,
                            lifParams, lifInit);

  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>("EE", SPARSE, GLOBALG, NO_DELAY,
                             "E", "E",
                             {}, excitatoryStaticSynapseInit,
                             excitatoryExpCurrParams, {});
  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>("EI", SPARSE, GLOBALG, NO_DELAY,
                             "E", "I",
                             {}, excitatoryStaticSynapseInit,
                             excitatoryExpCurrParams, {});
  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>("II", SPARSE, GLOBALG, NO_DELAY,
                             "I", "I",
                             {}, inhibitoryStaticSynapseInit,
                             inhibitoryExpCurrParams, {});
  model.addSynapsePopulation<Vogels2011, ExpCurr>("IE", SPARSE, INDIVIDUALG, NO_DELAY,
                             "I", "E",
                             vogels2011AdditiveSTDPParams, vogels2011AdditiveSTDPInit,
                             inhibitoryExpCurrParams, {});

  /*model.addSynapsePopulation("IE", NSYNAPSE, SPARSE, INDIVIDUALG, NO_DELAY, MY_EXP_CURR,
                             "I", "E",
                             staticSynapseInit, NULL,
                             NULL, inhibitoryExpCurrParams);*/
  /*model.setSpanTypeToPre("EE");
  model.setSpanTypeToPre("EI");
  model.setSpanTypeToPre("II");
  model.setSpanTypeToPre("IE");*/

  // Use zero-copy for spikes and weights as we want to record them every timestep
  model.setNeuronSpikeZeroCopy("E");
  model.setSynapseWeightUpdateVarZeroCopy("IE", "g");

  model.finalize();
}