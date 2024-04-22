#include "modelSpec.h"

#include "parameters.h"

//----------------------------------------------------------------------------
// RMaxProp
//----------------------------------------------------------------------------
class RMaxProp : public CustomUpdateModels::Base
{
    DECLARE_SNIPPET(RMaxProp);

    SET_UPDATE_CODE(
        "// Get gradients\n"
        "const scalar gradient = $(m) / $(updateTimesteps);\n"
        "// Calculate learning rate r\n"
        "$(upsilon) = fmax($(upsilon) * $(expRMS), gradient * gradient);\n"
        "const scalar r = *$(r0) / (sqrt($(upsilon)) + $(epsilon));\n"
        "// Update synaptic parameter\n"
        "$(variable) += r * gradient;\n"
        "$(variable) = fmin($(wMax), fmax($(wMin), $(variable)));\n"
        "$(m) = 0.0;\n");

    SET_EXTRA_GLOBAL_PARAMS({{"r0", "scalar*"}})
    
    SET_PARAM_NAMES({"updateTime", "tauRMS", "epsilon", "wMin", "wMax"});
    SET_DERIVED_PARAMS({
        {"updateTimesteps", [](const ParamValues &pars, double dt){ return pars.at("updateTime") / dt; }},
        {"expRMS", [](const ParamValues &pars, double){ return std::exp(-pars.at("updateTime") / pars.at("tauRMS")); }}});
    SET_VARS({{"upsilon", "scalar"}});
    SET_VAR_REFS({{"m", "scalar", VarAccessMode::READ_WRITE}, 
                  {"variable", "scalar", VarAccessMode::READ_WRITE}});
};
IMPLEMENT_SNIPPET(RMaxProp);

//----------------------------------------------------------------------------
// SuperSpikeBase
//----------------------------------------------------------------------------
class SuperSpikeBase : public WeightUpdateModels::Base
{
    SET_PARAM_NAMES({
        "tauRise",      // 0 - Rise time constant (ms)
        "tauDecay",     // 1 - Decay time constant (ms)
        "beta"});       // 2 - Beta

    SET_VARS({{"w", "scalar"}, {"e", "scalar"}, {"lambda", "scalar"},
              {"m", "scalar"}});
    SET_PRE_VARS({{"z", "scalar"}, {"zTilda", "scalar"}});
    SET_POST_VARS({{"sigmaPrime", "scalar"}});

    SET_SIM_CODE("$(addToInSyn, $(w));\n");

    SET_PRE_SPIKE_CODE("$(z) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE(
        "// filtered presynaptic trace\n"
        "$(z) += (-$(z) / $(tauRise)) * DT;\n"
        "$(zTilda) += ((-$(zTilda) + $(z)) / $(tauDecay)) * DT;\n");

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

//----------------------------------------------------------------------------
// SuperSpike
//----------------------------------------------------------------------------
class SuperSpike : public SuperSpikeBase
{
public:
    DECLARE_SNIPPET(SuperSpike);
    
    SET_SYNAPSE_DYNAMICS_CODE(
        "// Filtered eligibility trace\n"
        "$(e) += ($(zTilda) * $(sigmaPrime) - $(e) / $(tauRise))*DT;\n"
        "$(lambda) += ((-$(lambda) + $(e)) / $(tauDecay)) * DT;\n"
        "// Get error from neuron model and compute full \n"
        "// expression under integral and calculate m\n"
        "$(m) += $(lambda) * $(errTilda_post);\n");

};
IMPLEMENT_SNIPPET(SuperSpike);

//----------------------------------------------------------------------------
// SuperSpikeApprox
//----------------------------------------------------------------------------
class SuperSpikeApprox : public SuperSpikeBase
{
public:
    DECLARE_SNIPPET(SuperSpikeApprox);
    
    SET_EVENT_THRESHOLD_CONDITION_CODE("$(zTilda) > 1.0E-4");
    SET_EVENT_CODE(
        "// Filtered eligibility trace\n"
        "$(e) += ($(zTilda) * $(sigmaPrime) - $(e) / $(tauRise))*DT;\n"
        "$(lambda) += ((-$(lambda) + $(e)) / $(tauDecay)) * DT;\n"
        "// Get error from neuron model and compute full \n"
        "// expression under integral and calculate m\n"
        "$(m) += $(lambda) * $(errTilda_post);\n");

};
IMPLEMENT_SNIPPET(SuperSpikeApprox);

//----------------------------------------------------------------------------
// Feedback
//----------------------------------------------------------------------------
class Feedback : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(Feedback);

    SET_VARS({{"w", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(w) * $(errTilda_pre));\n");
};
IMPLEMENT_SNIPPET(Feedback);

//---------------------------------------------------------------------------
// FeedbackPSM
//---------------------------------------------------------------------------
//! Simple postsynaptic model which transfer input directly to neuron without any dynamics
class FeedbackPSM : public PostsynapticModels::Base
{
public:
    DECLARE_SNIPPET(FeedbackPSM);

    SET_APPLY_INPUT_CODE(
        "$(ISynFeedback) += $(inSyn);\n"
        "$(inSyn) = 0;\n");
};
IMPLEMENT_SNIPPET(FeedbackPSM);

//----------------------------------------------------------------------------
// Hidden
//----------------------------------------------------------------------------
class Hidden : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(Hidden);

    SET_PARAM_NAMES({
        "C",            // 0 - Membrane capacitance
        "tauMem",       // 1 - Membrane time constant (ms)
        "Vrest",        // 2 - Resting membrane voltage (mV)
        "Vthresh",      // 3 - Spiking threshold (mV)
        "tauRefrac"});  // 4 - Refractory time constant (ms)

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauMem")); }},
        {"Rmembrane", [](const ParamValues &pars, double){ return  pars.at("tauMem") / pars.at("C"); }}});

    SET_VARS({{"V", "scalar"}, {"refracTime", "scalar"}, {"errTilda", "scalar"}});

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
IMPLEMENT_SNIPPET(Hidden);

//----------------------------------------------------------------------------
// Output
//----------------------------------------------------------------------------
class Output : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(Output);

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
        {"ExpTC", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauMem")); }},
        {"Rmembrane", [](const ParamValues &pars, double){ return  pars.at("tauMem") / pars.at("C"); }},
        {"normFactor", [](const ParamValues&pars, double){ return 1.0 / (-std::exp(-calcTPeak(pars.at("tauRise"), pars.at("tauDecay")) / pars.at("tauRise")) + std::exp(-calcTPeak(pars.at("tauRise"), pars.at("tauDecay")) / pars.at("tauDecay"))); }},
        {"tRiseMult", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauRise")); }},
        {"tDecayMult", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauDecay")); }},
        {"tPeak", [](const ParamValues &pars, double){ return calcTPeak(pars.at("tauRise"), pars.at("tauDecay")); }},
        {"mulAvgErr", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("tauAvgErr")); }}});

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
IMPLEMENT_SNIPPET(Output);

void modelDefinition(NNmodel &model)
{
    model.setDT(Parameters::timestepMs);
    model.setName("superspike_demo");
    model.setTiming(true);

    //------------------------------------------------------------------------
    // Input layer parameters
    //------------------------------------------------------------------------
    VarValues inputVars{{"startSpike", uninitialisedVar()}, {"endSpike", uninitialisedVar()}};

    //------------------------------------------------------------------------
    // Hidden layer parameters
    //------------------------------------------------------------------------
    ParamValues hiddenParams{{"C", 10.0}, {"tauMem", 10.0}, {"Vrest", -60.0}, 
                             {"Vthresh", -50.0}, {"tauRefrac", 5.0}};
    
    
    VarValues hiddenVars{{"V", -60.0}, {"refracTime", 0.0}, {"errTilda", 0.0}};
        
    //------------------------------------------------------------------------
    // Output layer parameters
    //------------------------------------------------------------------------
    ParamValues outputParams{{"C", 10.0}, {"tauMem", 10.0}, {"Vrest", -60.0},
                             {"Vthresh", -50.0}, {"tauRefrac", 5.0}, {"tauRise", Parameters::tauRise},
                             {"tauDecay", Parameters::tauDecay}, {"tauAvgErr", Parameters::tauAvgErr}};
    VarValues outputVars{{"V", -60.0}, {"refracTime", 0.0}, {"errRise", 0.0}, {"errTilda", 0.0}, {"avgSqrErr", 0.0},
                         {"errDecay", 0.0}, {"startSpike", uninitialisedVar()}, {"endSpike", uninitialisedVar()}};

    //------------------------------------------------------------------------
    // Synapse parameters
    //------------------------------------------------------------------------
    ParamValues expCurrParams{{"tau", 5.0}};

    ParamValues superSpikeParams{{"tauRise", Parameters::tauRise}, 
                                 {"tauDecay", Parameters::tauDecay}, {"beta", 1000.0}};
    VarValues superSpikePreVars{{"z", 0.0}, {"zTilda", 0.0}};
    VarValues superSpikePostVars{{"sigmaPrime", 0.0}};

    ParamValues inputHiddenWeightDist{{"mean", 0.0}, {"sd", Parameters::w0 / std::sqrt((double)Parameters::numInput)},
                                      {"min", Parameters::wMin}, {"max", Parameters::wMax}};
    
    VarValues inputHiddenVars{{"w", initVar<InitVarSnippet::NormalClipped>(inputHiddenWeightDist)},
                              {"e", 0.0}, {"lambda", 0.0}, {"m", 0.0}};

    ParamValues hiddenOutputWeightDist{{"mean", 0.0}, {"sd", Parameters::w0 / std::sqrt((double)Parameters::numHidden)},
                                       {"min", Parameters::wMin}, {"max", Parameters::wMax}};
    VarValues hiddenOutputVars{{"w", initVar<InitVarSnippet::NormalClipped>(hiddenOutputWeightDist)},
                               {"e", 0.0}, {"lambda", 0.0}, {"m", 0.0}};
    
    // **HACK** this is actually a nasty corner case for the initialisation rules
    // We really want this uninitialised as we are going to copy over transpose
    // But then initialiseSparse would copy over host values
    VarValues feedbackVars{{"w", 0.0}};

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
    auto *inputHidden = model.addSynapsePopulation<SuperSpike, PostsynapticModels::ExpCurr>(
        "Input_Hidden", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Input", "Hidden",
        superSpikeParams, inputHiddenVars, superSpikePreVars, superSpikePostVars,
        expCurrParams, {});

    auto *hiddenOutput = model.addSynapsePopulation<SuperSpike, PostsynapticModels::ExpCurr>(
        "Hidden_Output", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Hidden", "Output",
        superSpikeParams, hiddenOutputVars, superSpikePreVars, superSpikePostVars,
        expCurrParams, {});

    auto *outputHidden = model.addSynapsePopulation<Feedback, FeedbackPSM>(
        "Output_Hidden", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Output", "Hidden",
        {}, feedbackVars,
        {}, {});
        
    //---------------------------------------------------------------------------
    // Custom updates
    //---------------------------------------------------------------------------
    WUVarReferences transposeHiddenOutputVarReferences{{"variable", createWUVarRef(hiddenOutput, "w", outputHidden, "w")}};
    model.addCustomUpdate<CustomUpdateModels::Transpose>("InputHiddenWeightTranspose", "CalculateTranspose",
                                                         {}, {}, transposeHiddenOutputVarReferences);
    
    
    ParamValues rMaxPropParams{{"updateTime", Parameters::updateTimeMs}, {"tauRMS", Parameters::tauRMS}, 
                               {"epsilon", Parameters::epsilon}, {"wMin", Parameters::wMin}, {"wMax", Parameters::wMax}};
    
    VarValues rMaxPropVarValues{{"upsilon", 0.0}};
    WUVarReferences rMaxPropInputHiddenVarReferences{{"m", createWUVarRef(inputHidden, "m")},
                                                     {"variable", createWUVarRef(inputHidden, "w")}};
    model.addCustomUpdate<RMaxProp>("InputHiddenWeightOptimiser", "GradientLearn",
                                    rMaxPropParams, rMaxPropVarValues, rMaxPropInputHiddenVarReferences);
    
    WUVarReferences rMaxPropHiddenOutputVarReferences{{"m", createWUVarRef(hiddenOutput, "m")},
                                                      {"variable", createWUVarRef(hiddenOutput, "w", outputHidden, "w")}};
    model.addCustomUpdate<RMaxProp>("HiddenOutputWeightOptimiser", "GradientLearn",
                                    rMaxPropParams, rMaxPropVarValues, rMaxPropHiddenOutputVarReferences);
}
