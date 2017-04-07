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

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(1.0);
  model.setName("if_curr");

  //---------------------------------------------------------------------------
  // Build model
  //---------------------------------------------------------------------------
  // LIF model parameters
  ClosedFormLIF::ParamValues lifParamVals(1.0, 20.0, -70.0, -70.0, -51.0, 0.0, 2.0);
  ClosedFormLIF::VarValues lifInitVals(-70.0, 0.0);

  ExpCurr::ParamValues expCurrParams(5.0);
  
  WeightUpdateModels::StaticPulse::VarValues staticSynapseInitVals(1.0);
  
  // Create IF_curr neuron
  model.addNeuronPopulation<NeuronModels::SpikeSource>("Stim", 1, {}, {});
  model.addNeuronPopulation<ClosedFormLIF>("Excitatory", 1, lifParamVals, lifInitVals);

  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>("StimToExcitatory", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY, 
                                                                       "Stim", "Excitatory",
                                                                       {}, staticSynapseInitVals,
                                                                       expCurrParams, {});

  model.finalize();
}