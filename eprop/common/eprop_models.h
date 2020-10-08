#pragma once

//----------------------------------------------------------------------------
// Recurrent
//----------------------------------------------------------------------------
class Recurrent : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Recurrent, 3, 3);

    SET_PARAM_NAMES({
        "TauM",         // Membrane time constant [ms]
        "Vthresh",      // Spiking threshold [mV]
        "TauRefrac"});  // Refractory time constant [ms]

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"E", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});

    SET_ADDITIONAL_INPUT_VARS({{"IsynFeedback", "scalar", 0.0}});

    SET_SIM_CODE(
        "$(E) = $(IsynFeedback);\n"
        "$(V) = ($(Alpha) * $(V)) + $(Isyn);\n"
        "if ($(RefracTime) > 0.0) {\n"
        "  $(RefracTime) -= DT;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(RefracTime) = $(TauRefrac);\n"
        "$(V) -= $(Vthresh);\n");

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Recurrent);

//----------------------------------------------------------------------------
// RecurrentALIF
//----------------------------------------------------------------------------
class RecurrentALIF : public NeuronModels::Base
{
public:
    DECLARE_MODEL(RecurrentALIF, 5, 4);

    SET_PARAM_NAMES({
        "TauM",         // Membrane time constant [ms]
        "TauAdap",      // Adaption time constant [ms]
        "Vthresh",      // Spiking threshold [mV]
        "TauRefrac",    // Refractory time constant [ms]
        "Beta"});       // Scale of adaption [mV]

    SET_VARS({{"V", "scalar"}, {"A", "scalar"}, {"RefracTime", "scalar"}, {"E", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"Rho", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }}});

    SET_ADDITIONAL_INPUT_VARS({{"IsynFeedback", "scalar", 0.0}});

    SET_SIM_CODE(
        "$(E) = $(IsynFeedback);\n"
        "$(V) = ($(Alpha) * $(V)) + $(Isyn);\n"
        "$(A) *= $(Rho);\n"
        "if ($(RefracTime) > 0.0) {\n"
        "  $(RefracTime) -= DT;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= ($(Vthresh) + ($(Beta) * $(A)))");

    SET_RESET_CODE(
        "$(RefracTime) = $(TauRefrac);\n"
        "$(V) -= $(Vthresh);\n"
        "$(A) += 1.0;\n");

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(RecurrentALIF);

//---------------------------------------------------------------------------
// Feedback
//---------------------------------------------------------------------------
//! Simple postsynaptic model which transfer input directly to neuron without any dynamics
class Feedback : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(Feedback, 0, 0);

    SET_APPLY_INPUT_CODE(
        "$(IsynFeedback) += $(inSyn);\n"
        "$(inSyn) = 0;\n");
};
IMPLEMENT_MODEL(Feedback);

//---------------------------------------------------------------------------
// Continuous
//---------------------------------------------------------------------------
//! Simple continous synapse for error feedback
class Continuous : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Continuous, 0, 1);

    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(g) * $(E_pre));\n");
};
IMPLEMENT_MODEL(Continuous);

//---------------------------------------------------------------------------
// EProp
//---------------------------------------------------------------------------
//! Basic implementation of EProp learning rule
class EProp : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(EProp, 4, 5, 1, 2);
    
    SET_PARAM_NAMES({"TauE",        // Eligibility trace time constant [ms]
                     "CReg",        // Regularizer strength
                     "FTarget",     // Target spike rate [Hz]
                     "TauFAvg"});   // Firing rate averaging time constant [ms]

    SET_VARS({{"g", "scalar"}, {"eFiltered", "scalar"}, {"DeltaG", "scalar"},
              {"M", "scalar"}, {"V", "scalar"}});

    SET_PRE_VARS({{"ZFilter", "scalar"}});
    SET_POST_VARS({{"Psi", "scalar"}, {"FAvg", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"FTargetTimestep", [](const std::vector<double> &pars, double dt){ return pars[2] / (1000.0 * dt); }},
        {"AlphaFAv", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[3]); }}});

    SET_SIM_CODE("$(addToInSyn, $(g));\n");

    SET_SYNAPSE_DYNAMICS_CODE(
        "const scalar e = $(ZFilter) * $(Psi);\n"
        "scalar eFiltered = $(eFiltered);\n"
        "eFiltered = (eFiltered * $(Alpha)) + e;\n"
        "$(DeltaG) += (eFiltered * $(E_post)) - (($(FTargetTimestep) - $(FAvg)) * $(CReg) * e);\n"
        "$(eFiltered) = eFiltered;\n");

    SET_PRE_SPIKE_CODE("$(ZFilter) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("$(ZFilter) *= $(Alpha);\n");
    
    SET_POST_SPIKE_CODE("$(FAvg) += (1.0 - $(AlphaFAv));\n");
    SET_POST_DYNAMICS_CODE(
        "$(FAvg) *= $(AlphaFAv);\n"
        "if ($(RefracTime_post) > 0.0) {\n"
        "  $(Psi) = 0.0;\n"
        "}\n"
        "else {\n"
        "  $(Psi) = (1.0 / $(Vthresh_post)) * 0.3 * fmax(0.0, 1.0 - fabs(($(V_post) - $(Vthresh_post)) / $(Vthresh_post)));\n"
        "}\n");
};
IMPLEMENT_MODEL(EProp);

//---------------------------------------------------------------------------
// EPropALIF
//---------------------------------------------------------------------------
//! Basic implementation of EProp learning rule
class EPropALIF : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(EPropALIF, 6, 6, 1, 2);
    
    SET_PARAM_NAMES({"TauE",        // Eligibility trace time constant [ms]
                     "TauA",        // Neuron adaption time constant [ms]
                     "CReg",        // Regularizer strength
                     "FTarget",     // Target spike rate [Hz]
                     "TauFAvg",     // Firing rate averaging time constant [ms]
                     "Beta"});      // Scale of neuron adaption [mV]

    SET_VARS({{"g", "scalar"}, {"eFiltered", "scalar"}, {"epsilonA", "scalar"}, 
              {"DeltaG", "scalar"}, {"M", "scalar"}, {"V", "scalar"}});

    SET_PRE_VARS({{"ZFilter", "scalar"}});
    SET_POST_VARS({{"Psi", "scalar"}, {"FAvg", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"Rho", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"FTargetTimestep", [](const std::vector<double> &pars, double dt){ return pars[3] / (1000.0 * dt); }},
        {"AlphaFAv", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[4]); }}});

    SET_SIM_CODE("$(addToInSyn, $(g));\n");

    SET_SYNAPSE_DYNAMICS_CODE(
        "// Calculate some common factors in e and epsilon update\n"
        "scalar epsilonA = $(epsilonA);\n"
        "const scalar psiZFilter = $(Psi) * $(ZFilter);\n"
        "const scalar psiBetaEpsilonA = $(Psi) * $(Beta) * epsilonA;\n"
        "// Calculate e and episilonA\n"
        "const scalar e = psiZFilter  - psiBetaEpsilonA;\n"
        "$(epsilonA) = psiZFilter + (($(Rho) * epsilonA) - psiBetaEpsilonA);\n"
        "// Calculate filtered version of eligibility trace\n"
        "scalar eFiltered = $(eFiltered);\n"
        "eFiltered = (eFiltered * $(Alpha)) + e;\n"
        "// Apply weight update\n"
        "$(DeltaG) += (eFiltered * $(E_post)) - (($(FTargetTimestep) - $(FAvg)) * $(CReg) * e);\n"
        "$(eFiltered) = eFiltered;\n");

    SET_PRE_SPIKE_CODE("$(ZFilter) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("$(ZFilter) *= $(Alpha);\n");
    
    SET_POST_SPIKE_CODE("$(FAvg) += (1.0 - $(AlphaFAv));\n");
    SET_POST_DYNAMICS_CODE(
        "$(FAvg) *= $(AlphaFAv);\n"
        "if ($(RefracTime_post) > 0.0) {\n"
        "  $(Psi) = 0.0;\n"
        "}\n"
        "else {\n"
        "  $(Psi) = (1.0 / $(Vthresh_post)) * 0.3 * fmax(0.0, 1.0 - fabs(($(V_post) - ($(Vthresh_post) + ($(Beta_post) * $(A_post)))) / $(Vthresh_post)));\n"
        "}\n");
};
IMPLEMENT_MODEL(EPropALIF);

//---------------------------------------------------------------------------
// OutputLearning
//---------------------------------------------------------------------------
//! Basic implementation of output learning rule
class OutputLearning : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(OutputLearning, 1, 4, 1, 0);

    SET_PARAM_NAMES({"TauE"});  // Eligibility trace time constant [ms]

    SET_VARS({{"g", "scalar"}, {"DeltaG", "scalar"},
              {"M", "scalar"}, {"V", "scalar"}});

    SET_PRE_VARS({{"ZFilter", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});

    SET_SIM_CODE("$(addToInSyn, $(g));\n");

    SET_SYNAPSE_DYNAMICS_CODE("$(DeltaG) += $(ZFilter) * $(E_post);\n");

    SET_PRE_SPIKE_CODE("$(ZFilter) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("$(ZFilter) *= $(Alpha);\n");
};
IMPLEMENT_MODEL(OutputLearning);
