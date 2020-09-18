#include "modelSpec.h"

#include "parameters.h"

//----------------------------------------------------------------------------
// SuperSpike
//----------------------------------------------------------------------------
class SuperSpike : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(SuperSpike, 2, 5);

    SET_PARAM_NAMES({
        "tauRise",      // 0 - Rise time constant (ms)
        "tauDecay"});   // 1 - Decay time constant (ms)

    SET_VARS({{"w", "scalar"}, {"e", "scalar"}, {"lambda", "scalar"},
              {"upsilon", "scalar"}, {"m", "scalar"}});

    SET_SIM_CODE("$(addToInSyn, $(g));\n");

    SET_LEARN_POST_CODE(
        "// Filtered eligibility trace\n"
        "$(e) += ($(z_tilda_pre) * $(sigma_prime_post) - $(e)/$(tauRise))*DT;\n"
        "$(lambda) += ((-$(lambda) + $(e)) / $(tauDecay)) * DT;\n"
        "// Get error from neuron model and compute full expression\n"
        "// under integral and calculate m\n"
        "$(m) += $(lambda) * $(err_tilda_post);\n");
};
IMPLEMENT_MODEL(SuperSpike);

//----------------------------------------------------------------------------
// Input
//----------------------------------------------------------------------------
class Input : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Input, 2, 4);

    SET_PARAM_NAMES({
        "tauRise",      // 0 - Rise time constant (ms)
        "tauDecay"});    // 1 - Decay time constant

    SET_VARS({{"startSpike", "unsigned int"}, {"endSpike", "unsigned int"}, {"z", "scalar"}, {"zTilda", "scalar"}});

    SET_EXTRA_GLOBAL_PARAMS({{"spikeTimes", "scalar*"}});

    SET_SIM_CODE(
        "// filtered presynaptic trace\n"
        "$(z) += (-$(z) / $(tauRise)) * DT;\n"
        "$(zTilda) += ((-$(zTilda) + $(z)) / $(tauDecay)) * DT;\n"
        "if ($(zTilda) < 0.0000001) {\n"
        "    $(zTilda) = 0.0;\n"
        "}\n");
    SET_RESET_CODE(
        "$(startSpike)++;\n"
        "$(z) += 1.0;\n");
    SET_THRESHOLD_CONDITION_CODE("$(startSpike) != $(endSpike) && $(t)>= $(spikeTimes)[$(startSpike)]");

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Input);


//----------------------------------------------------------------------------
// Hidden
//----------------------------------------------------------------------------
class Hidden : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Hidden, 7, 5);

    SET_PARAM_NAMES({
        "C",            // 0 - Membrane capacitance
        "tauMem",       // 1 - Membrane time constant (ms)
        "Vrest",        // 2 - Resting membrane voltage (mV)
        "Vthresh",      // 3 - Spiking threshold (mV)
        "tauRefrac",    // 4 - Refractory time constant (ms)
        "tauRise",      // 5 - Rise time constant (ms)
        "tauDecay"});   // 6 - Decay time constant

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"refracTime", "scalar"}, {"errTilda", "scalar"},
              {"z", "scalar"}, {"zTilda", "scalar"}});

    SET_EXTRA_GLOBAL_PARAMS({{"spikeTimes", "scalar*"}});

    SET_ADDITIONAL_INPUT_VARS({{"ISynFeedback", "scalar", 0.0}});
    SET_SIM_CODE(
        "// membrane potential dynamics\n"
        "if ($(refracTime) == $(tauRefrac)) {\n"
        "    $(V) = $(Vrest);\n"
        "}\n"
        "if ($(refracTime) <= 0.0) {\n"
        "    scalar alpha = ($(Isyn) * $(Rmembrane)) + $(Vrest);\n"
        "    $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else {\n"
        "    $(refracTime) -= DT;\n"
        "}\n"
        "// filtered presynaptic trace\n"
        "$(z) += (-$(z) / $(tauRise)) * DT;\n"
        "$(zTilda) += ((-$(zTilda) + $(z)) / $(tauDecay)) * DT;\n"
        "if ($(zTilda) < 0.0000001) {\n"
        "    $(zTilda) = 0.0;\n"
        "}\n"
        "// error\n"
        "$(errTilda) = $(ISynFeedback);\n");

    SET_RESET_CODE(
        "$(refracTime) = $(tauRefrac);\n"
        "$(z) += 1.0;\n");

    SET_THRESHOLD_CONDITION_CODE("$(refracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Hidden);

//----------------------------------------------------------------------------
// Output
//----------------------------------------------------------------------------
class Output : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Output, 7, 7);

    SET_PARAM_NAMES({
        "C",            // 0 - Membrane capacitance
        "tauMem",       // 1 - Membrane time constant (ms)
        "Vrest",        // 2 - Resting membrane voltage (mV)
        "Vthresh",      // 3 - Spiking threshold (mV)
        "tauRefrac",    // 4 - Refractory time constant (ms)
        "tauRise",      // 5 - Rise time constant (ms)
        "tauDecay"});   // 6 - Decay time constant

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }},
        {"normFactor", [](const std::vector<double> &pars, double){ return (1.0 / (-std::exp(-pars[7] / pars[5])) + std::exp(-pars[7] / pars[6])); }},
        {"tRiseMult", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[5]); }},
        {"tDecayMult", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[6]); }},
        {"tPeak", [](const std::vector<double> &pars, double){ return ((pars[6] * pars[5]) / (pars[6] - pars[5])) * std::log(pars[6] / pars[5]); }}});

    SET_VARS({{"V", "scalar"}, {"refracTime", "scalar"}, {"errRise", "scalar"}, {"errTilda", "scalar"},
              {"errDecay", "scalar"}, {"startSpike", "unsigned int"}, {"endSpike", "unsigned int"}});
    SET_EXTRA_GLOBAL_PARAMS({{"spikeTimes", "scalar*"}});

    SET_SIM_CODE(
        "// membrane potential dynamics\n"
        "if ($(refracTime) == $(tauRefrac)) {\n"
        "    $(V) = $(Vrest);\n"
        "}\n"
        "if ($(refracTime) <= 0.0) {\n"
        "    scalar alpha = ($(Isyn) * $(Rmembrane)) + $(Vrest);\n"
        "    $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else {\n"
        "    $(refracTime) -= DT;\n"
        "}\n"
        "// error\n"
        "scalar sPred = 0.0;\n"
        "if ($(startSpike) != $(endSpike) && $(t) >= $(spikeTimes)[$(startSpike)]) {\n"
        "    $(startSpike)++;\n"
        "    sPred = 1.0;\n"
        "}\n"
        "const scalar sReal = $(refracTime) <= 0.0 && $(V) >= $(Vthresh) ? 1.0 : 0.0;\n"
        "const scalar mismatch = sPred - sReal;\n"
        "$(errRise) = ($(errRise) * $(tRiseMult)) + mismatch;\n"
        "$(errDecay) = ($(errDecay) * $(tDecayMult)) + mismatch;\n"
        "$(errTilda) = ($(errDecay) - $(errRise)) * $(normFactor);\n");

    SET_RESET_CODE("$(refracTime) = $(tauRefrac);\n");

    SET_THRESHOLD_CONDITION_CODE("$(refracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Output);

void modelDefinition(NNmodel &model)
{
    model.setDT(0.1);
    model.setName("superspike_demo");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTiming(true);

    //------------------------------------------------------------------------
    // Input layer
    //------------------------------------------------------------------------
    Input::ParamValues inputParams(
        Parameters::tauRise,    // 0 - Rise time constant (ms)
        Parameters::tauDecay);  // 1 - Decay time constant (ms)

    Input::VarValues inputVars(
        uninitialisedVar(),     // 0 - startSpike
        uninitialisedVar(),     // 1 - endSpike
        0.0,                    // 2 - z
        0.0);                   // 3 - zTilda

    //------------------------------------------------------------------------
    // Hidden layer
    //------------------------------------------------------------------------
    Hidden::ParamValues hiddenParams(
        10.0,                   // 0 - Membrane capacitance
        10.0,                   // 1 - Membrane time constant (ms)
        -60.0,                  // 2 - Resting membrane voltage (mV)
        -50.0,                  // 3 - Spiking threshold (mV)
        5.0,                    // 4 - Refractory time constant (ms)
        Parameters::tauRise,    // 5 - Rise time constant (ms)
        Parameters::tauDecay);  // 6 - Decay time constant

    Hidden::VarValues hiddenVars(
        -60.0,  // V
        0.0,    // refracTime
        0.0,    // errTilda
        0.0,    // z
        0.0);   // zTilda

    //------------------------------------------------------------------------
    // Output layer
    //------------------------------------------------------------------------
    Output::ParamValues outputParams(
        10.0,                   // 0 - Membrane capacitance
        10.0,                   // 1 - Membrane time constant (ms)
        -60.0,                  // 2 - Resting membrane voltage (mV)
        -50.0,                  // 3 - Spiking threshold (mV)
        5.0,                    // 4 - Refractory time constant (ms)
        Parameters::tauRise,    // 5 - Rise time constant (ms)
        Parameters::tauDecay);  // 6 - Decay time constant
    Output::VarValues outputVars(
        -60.0,                  // V
        0.0,                    // refracTime
        0.0,                    // errRise
        0.0,                    // errTilda
        0.0,                    // errDecay
        uninitialisedVar(),     // startSpike
        uninitialisedVar());    // endSpike

    model.addNeuronPopulation<Input>("Input", Parameters::numInput, inputParams, inputVars);
    model.addNeuronPopulation<Hidden>("Hidden", Parameters::numHidden, hiddenParams, hiddenVars);
    model.addNeuronPopulation<Output>("Output", Parameters::numOutput, outputParams, outputVars);
}
