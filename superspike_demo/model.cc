#include "modelSpec.h"

#include "parameters.h"

//----------------------------------------------------------------------------
// SuperSpike
//----------------------------------------------------------------------------
class SuperSpike : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(SuperSpike, 3, 5, 2, 1);

    SET_PARAM_NAMES({
        "tauRise",      // 0 - Rise time constant (ms)
        "tauDecay",     // 1 - Decay time constant (ms)
        "beta"});       // 2 - Beta

    SET_VARS({{"w", "scalar"}, {"e", "scalar"}, {"lambda", "scalar"},
              {"upsilon", "scalar"}, {"m", "scalar"}});
    SET_PRE_VARS({{"z", "scalar"}, {"zTilda", "scalar"}});
    SET_POST_VARS({{"sigmaPrime", "scalar"}});

    SET_SIM_CODE("$(addToInSyn, $(w));\n");

    SET_SYNAPSE_DYNAMICS_CODE(
        "// Filtered eligibility trace\n"
        "$(e) += ($(zTilda) * $(sigmaPrime) - $(e) / $(tauRise))*DT;\n"
        "$(lambda) += ((-$(lambda) + $(e)) / $(tauDecay)) * DT;\n"
        "// Get error from neuron model and compute full \n"
        "// expression under integral and calculate m\n"
        "$(m) += $(lambda) * $(errTilda_post);\n");

    SET_PRE_SPIKE_CODE("$(z) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE(
        "// filtered presynaptic trace\n"
        "$(z) += (-$(z) / $(tauRise)) * DT;\n"
        "$(zTilda) += ((-$(zTilda) + $(z)) / $(tauDecay)) * DT;\n"
        "if ($(zTilda) < 0.0000001) {\n"
        "    $(zTilda) = 0.0;\n"
        "}\n");

    SET_POST_DYNAMICS_CODE(
        "// filtered partial derivative\n"
        "if($(V_post) < -80.0) {\n"
        "   $(sigmaPrime) = 0.0;\n"
        "}\n"
        "else {\n"
        "   const scalar onePlusHi = 1.0 + fabs($(beta) * 0.001 * ($(V_post) - $(Vthresh_post)));\n"
        "   $(sigmaPrime) = $(beta) / (onePlusHi * onePlusHi);\n"
        "}\n");

};
IMPLEMENT_MODEL(SuperSpike);

//----------------------------------------------------------------------------
// Feedback
//----------------------------------------------------------------------------
class Feedback : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Feedback, 0, 1);

    SET_VARS({{"w", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(w) * $(errTilda_pre));\n");
};
IMPLEMENT_MODEL(Feedback);

//---------------------------------------------------------------------------
// FeedbackPSM
//---------------------------------------------------------------------------
//! Simple postsynaptic model which transfer input directly to neuron without any dynamics
class FeedbackPSM : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(FeedbackPSM, 0, 0);

    SET_APPLY_INPUT_CODE(
        "$(ISynFeedback) += $(inSyn);\n"
        "$(inSyn) = 0;\n");
};
IMPLEMENT_MODEL(FeedbackPSM);

//----------------------------------------------------------------------------
// Hidden
//----------------------------------------------------------------------------
class Hidden : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Hidden, 5, 3);

    SET_PARAM_NAMES({
        "C",            // 0 - Membrane capacitance
        "tauMem",       // 1 - Membrane time constant (ms)
        "Vrest",        // 2 - Resting membrane voltage (mV)
        "Vthresh",      // 3 - Spiking threshold (mV)
        "tauRefrac"});  // 4 - Refractory time constant (ms)

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"refracTime", "scalar"}, {"errTilda", "scalar"}});

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
        "// error\n"
        "$(errTilda) = $(ISynFeedback);\n");

    SET_RESET_CODE(
        "$(refracTime) = $(tauRefrac);\n");

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
    DECLARE_MODEL(Output, 8, 8);

    SET_PARAM_NAMES({
        "C",            // 0 - Membrane capacitance
        "tauMem",       // 1 - Membrane time constant (ms)
        "Vrest",        // 2 - Resting membrane voltage (mV)
        "Vthresh",      // 3 - Spiking threshold (mV)
        "tauRefrac",    // 4 - Refractory time constant (ms)
        "tauRise",      // 5 - Rise time constant (ms)
        "tauDecay",     // 6 - Decay time constant (ms)
        "tauAvgErr"});  // 7 - Average error time constant (ms)

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }},
        {"normFactor", [](const std::vector<double> &pars, double){ return 1.0 / (-std::exp(-calcTPeak(pars[5], pars[6]) / pars[5]) + std::exp(-calcTPeak(pars[5], pars[6]) / pars[6])); }},
        {"tRiseMult", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[5]); }},
        {"tDecayMult", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[6]); }},
        {"tPeak", [](const std::vector<double> &pars, double){ return calcTPeak(pars[5], pars[6]); }},
        {"mulAvgErr", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[7]); }}});

    SET_VARS({{"V", "scalar"}, {"refracTime", "scalar"}, {"errRise", "scalar"}, {"errTilda", "scalar"}, {"avgSqrErr", "scalar"},
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
        "const scalar sReal = ($(refracTime) <= 0.0 && $(V) >= $(Vthresh)) ? 1.0 : 0.0;\n"
        "const scalar mismatch = sPred - sReal;\n"
        "$(errRise) = ($(errRise) * $(tRiseMult)) + mismatch;\n"
        "$(errDecay) = ($(errDecay) * $(tDecayMult)) + mismatch;\n"
        "$(errTilda) = ($(errDecay) - $(errRise)) * $(normFactor);\n"
        "// calculate average error trace\n"
        "const scalar temp = $(errTilda) * $(errTilda) * DT * 0.001;\n"
        "$(avgSqrErr) *= $(mulAvgErr);\n"
        "$(avgSqrErr) += temp;\n");

    SET_RESET_CODE("$(refracTime) = $(tauRefrac);\n");

    SET_THRESHOLD_CONDITION_CODE("$(refracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_NEEDS_AUTO_REFRACTORY(false);

private:
    static double calcTPeak(double tauRise, double tauDecay)
    {
        return ((tauDecay * tauRise) / (tauDecay - tauRise)) * std::log(tauDecay / tauRise);
    }
};
IMPLEMENT_MODEL(Output);

void modelDefinition(NNmodel &model)
{
    model.setDT(Parameters::timestepMs);
    model.setName("superspike_demo");
    model.setTiming(true);

    //------------------------------------------------------------------------
    // Input layer parameters
    //------------------------------------------------------------------------
    NeuronModels::SpikeSourceArray::VarValues inputVars(
        uninitialisedVar(),     // 0 - startSpike
        uninitialisedVar());    // 1 - endSpike

    //------------------------------------------------------------------------
    // Hidden layer parameters
    //------------------------------------------------------------------------
    Hidden::ParamValues hiddenParams(
        10.0,                   // 0 - Membrane capacitance
        10.0,                   // 1 - Membrane time constant (ms)
        -60.0,                  // 2 - Resting membrane voltage (mV)
        -50.0,                  // 3 - Spiking threshold (mV)
        5.0);                   // 4 - Refractory time constant (ms)

    Hidden::VarValues hiddenVars(
        -60.0,  // V
        0.0,    // refracTime
        0.0);   // errTilda

    //------------------------------------------------------------------------
    // Output layer parameters
    //------------------------------------------------------------------------
    Output::ParamValues outputParams(
        10.0,                   // 0 - Membrane capacitance
        10.0,                   // 1 - Membrane time constant (ms)
        -60.0,                  // 2 - Resting membrane voltage (mV)
        -50.0,                  // 3 - Spiking threshold (mV)
        5.0,                    // 4 - Refractory time constant (ms)
        Parameters::tauRise,    // 5 - Rise time constant (ms)
        Parameters::tauDecay,   // 6 - Decay time constant (ms)
        Parameters::tauAvgErr); // 7 - Average error time constant (ms)
    Output::VarValues outputVars(
        -60.0,                  // V
        0.0,                    // refracTime
        0.0,                    // errRise
        0.0,                    // errTilda
        0.0,                    // errDecay
        0.0,                    // avgSqrErr
        uninitialisedVar(),     // startSpike
        uninitialisedVar());    // endSpike

    //------------------------------------------------------------------------
    // Synapse parameters
    //------------------------------------------------------------------------
    PostsynapticModels::ExpCurr::ParamValues expCurrParams(
        5.0);  // 0 - TauSyn (ms)

    SuperSpike::ParamValues superSpikeParams(
        Parameters::tauRise,    // 0 - Rise time constant (ms)
        Parameters::tauDecay,   // 1 - Decay time constant (ms)
        1000.0);                // 2 - Beta

    SuperSpike::PreVarValues superSpikePreVars(
        0.0,    // z
        0.0);   // zTilda

    SuperSpike::PostVarValues superSpikePostVars(
        0.0);   // sigmaPrime

    InitVarSnippet::NormalClipped::ParamValues inputHiddenWeightDist(
        0.0,                                                        // 0 - mean
        Parameters::w0 / std::sqrt((double)Parameters::numInput),   // 1 - standard deviation
        Parameters::wMin,                                           // 2 - min
        Parameters::wMax);                                          // 3 - max
    SuperSpike::VarValues inputHiddenVars(
        initVar<InitVarSnippet::NormalClipped>(inputHiddenWeightDist),  // w
        0.0,                                                            // e
        0.0,                                                            // lambda
        0.0,                                                            // upsilon
        0.0);                                                           // m

    InitVarSnippet::NormalClipped::ParamValues hiddenOutputWeightDist(
        0.0,                                                        // 0 - mean
        Parameters::w0 / std::sqrt((double)Parameters::numHidden),  // 1 - standard deviation
        Parameters::wMin,                                           // 2 - min
        Parameters::wMax);                                          // 3 - max
    SuperSpike::VarValues hiddenOutputVars(
        initVar<InitVarSnippet::NormalClipped>(hiddenOutputWeightDist), // w
        0.0,                                                            // e
        0.0,                                                            // lambda
        0.0,                                                            // upsilon
        0.0);                                                           // m

    // **HACK** this is actually a nasty corner case for the initialisation rules
    // We really want this uninitialised as we are going to copy over transpose
    // But then initialiseSparse would copy over host values
    Feedback::VarValues feedbackVars(
        0.0);

    //------------------------------------------------------------------------
    // Neuron groups
    //------------------------------------------------------------------------
    auto *input = model.addNeuronPopulation<NeuronModels::SpikeSourceArray>("Input", Parameters::numInput, {}, inputVars);
    auto *hidden = model.addNeuronPopulation<Hidden>("Hidden", Parameters::numHidden, hiddenParams, hiddenVars);
    auto *output = model.addNeuronPopulation<Output>("Output", Parameters::numOutput, outputParams, outputVars);

    input->setSpikeRecordingEnabled(true);
    hidden->setSpikeRecordingEnabled(true);
    output->setSpikeRecordingEnabled(true);

    //------------------------------------------------------------------------
    // Synapse groups
    //------------------------------------------------------------------------
    model.addSynapsePopulation<SuperSpike, PostsynapticModels::ExpCurr>(
        "Input_Hidden", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Input", "Hidden",
        superSpikeParams, inputHiddenVars, superSpikePreVars, superSpikePostVars,
        expCurrParams, {});

    model.addSynapsePopulation<SuperSpike, PostsynapticModels::ExpCurr>(
        "Hidden_Output", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Hidden", "Output",
        superSpikeParams, hiddenOutputVars, superSpikePreVars, superSpikePostVars,
        expCurrParams, {});

    model.addSynapsePopulation<Feedback, FeedbackPSM>(
        "Output_Hidden", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Output", "Hidden",
        {}, feedbackVars,
        {}, {});
}
