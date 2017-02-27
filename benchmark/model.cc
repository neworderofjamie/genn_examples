#include <cmath>
#include <vector>

#include "modelSpec.h"

#define NUM_PRE_NEURONS 80000
#define NUM_POST_NEURONS 32

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

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(1.0);
    model.setName("benchmark");


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
        0.0,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    // **TODO** uniform random
    ClosedFormLIF::InitValues lifInit(
        -55.0,  // 0 - V
        0.0);    // 1 - RefracTime

    NeuronModels::Poisson::ParamValues poissonParams(
        10.0,        // 0 - firing rate
        2.5,        // 1 - refratory period
        20.0,       // 2 - Vspike
        -60.0);       // 3 - Vrest

    NeuronModels::Poisson::InitValues poissonInit(
        -60.0,        // 0 - V
        0,           // 1 - seed
        -10.0);     // 2 - SpikeTime

  // Static synapse parameters
  WeightUpdateModels::StaticPulse::InitValues staticSynapseInit(
      0.00);    // 0 - Wij (nA)

  // Exponential current parameters
  ExpCurr::ParamValues expCurrParams(
      5.0);  // 0 - TauSyn (ms)

  // Create IF_curr neuron
  model.addNeuronPopulation<NeuronModels::Poisson>("Stim", NUM_PRE_NEURONS,
                            poissonParams, poissonInit);
  model.addNeuronPopulation<ClosedFormLIF>("Neurons", NUM_POST_NEURONS,
                            lifParams, lifInit);

  model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>("Syn", SPARSE, INDIVIDUALG, NO_DELAY,
                             "Stim", "Neurons",
                             {}, staticSynapseInit,
                             expCurrParams, {});

  /*model.setSpanTypeToPre("EE");*/
  model.finalize();
}