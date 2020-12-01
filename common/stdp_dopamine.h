#pragma once

// GeNN includes
#include "modelSpec.h"

//----------------------------------------------------------------------------
// STDPDopamine
//----------------------------------------------------------------------------
class STDPDopamine : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(STDPDopamine, 8, 2);

    SET_PARAM_NAMES({
        "tauPlus",  // 0 - Potentiation time constant (ms)
        "tauMinus", // 1 - Depression time constant (ms)
        "tauC",     // 2 - Synaptic tag time constant (ms)
        "tauD",     // 3 - Dopamine time constant (ms)
        "aPlus",    // 4 - Rate of potentiation
        "aMinus",   // 5 - Rate of depression
        "wMin",     // 6 - Minimum weight
        "wMax",     // 7 - Maximum weight
    });

    SET_VARS({
        {"g", "scalar"},    // Synaptic weight
        {"c", "scalar"}});  // Synaptic tag

    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "// Calculate how much tag has decayed since last update\n"
        "const scalar tc = fmax($(prev_sT_pre), fmax($(prev_sT_post), $(prev_seT_pre)));\n"
        "const scalar tagDT = $(t) - tc;\n"
        "const scalar tagDecay = exp(-tagDT / $(tauC));\n"
        "// Calculate how much dopamine has decayed since last update\n"
        "const scalar dopamineDT = $(t) - $(seT_pre);\n"
        "const scalar dopamineDecay = exp(-dopamineDT / $(tauD));\n"
        "// Calculate offset to integrate over correct area\n"
        "const scalar offset = (tc <= $(seT_pre)) ? exp(-($(seT_pre) - tc) / $(tauC)) : exp(-(tc - $(seT_pre)) / $(tauD));\n"
        "// Update weight and clamp\n"
        "$(g) += ($(c) * $(d) * $(scale)) * ((tagDecay * dopamineDecay) - offset);\n"
        "$(g) = fmax($(wMin), fmin($(wMax), $(g)));\n"
        "// Decay tag and apply STDP\n"
        "scalar newTag = $(c) * tagDecay;\n"
        "const scalar dt = $(t) - $(sT_post);\n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar timing = exp(-dt / $(tauMinus));\n"
        "    newTag -= ($(aMinus) * timing);\n"
        "}\n"
        "// Write back updated tag and update time\n"
        "$(c) = newTag;\n");

    SET_EVENT_CODE(
        "// Calculate how much tag has decayed since last update\n"
        "const scalar tc = fmax($(sT_pre), fmax($(prev_sT_post), $(prev_seT_pre)));\n"
        "const scalar tagDT = $(t) - tc;\n"
        "const scalar tagDecay = exp(-tagDT / $(tauC));\n"
        "// Calculate how much dopamine has decayed since last update\n"
        "const scalar dopamineDT = $(t) - $(seT_pre);\n"
        "const scalar dopamineDecay = exp(-dopamineDT / $(tauD));\n"
        "// Calculate offset to integrate over correct area\n"
        "const scalar offset = (tc <= $(seT_pre)) ? exp(-($(seT_pre) - tc) / $(tauC)) : exp(-(tc - $(seT_pre)) / $(tauD));\n"
        "// Update weight and clamp\n"
        "$(g) += ($(c) * $(d) * $(scale)) * ((tagDecay * dopamineDecay) - offset);\n"
        "$(g) = fmax($(wMin), fmin($(wMax), $(g)));\n"
        "// Write back updated tag and update time\n"
        "$(c) *= tagDecay;\n");

    SET_LEARN_POST_CODE(
        "// Calculate how much tag has decayed since last update\n"
        "const scalar tc = fmax($(sT_pre), fmax($(prev_sT_post), $(seT_pre)));\n"
        "const scalar tagDT = $(t) - tc;\n"
        "const scalar tagDecay = exp(-tagDT / $(tauC));\n"
        "// Calculate how much dopamine has decayed since last update\n"
        "const scalar dopamineDT = $(t) - $(seT_pre);\n"
        "const scalar dopamineDecay = exp(-dopamineDT / $(tauD));\n"
        "// Calculate offset to integrate over correct area\n"
        "const scalar offset = (tc <= $(seT_pre)) ? exp(-($(seT_pre) - tc) / $(tauC)) : exp(-(tc - $(seT_pre)) / $(tauD));\n"
        "// Update weight and clamp\n"
        "$(g) += ($(c) * $(d) * $(scale)) * ((tagDecay * dopamineDecay) - offset);\n"
        "$(g) = max($(wMin), min($(wMax), $(g)));\n"
        "// Decay tag and apply STDP\n"
        "scalar newTag = $(c) * tagDecay;\n"
        "const scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar timing = exp(-dt / $(tauPlus));\n"
        "    newTag += ($(aPlus) * timing);\n"
        "}\n"
        "// Write back updated tag and update time\n"
        "$(c) = newTag;\n");

    SET_EVENT_THRESHOLD_CONDITION_CODE("$(injectDopamine)");

    SET_EXTRA_GLOBAL_PARAMS({
        {"injectDopamine", "bool"},
        {"d", "scalar"}});

    SET_DERIVED_PARAMS({
        {"scale", [](const std::vector<double> &pars, double){ return 1.0 / -((1.0 / pars[2]) + (1.0 / pars[3])); }}
    });

    SET_NEEDS_PRE_SPIKE_TIME(true);  
    SET_NEEDS_POST_SPIKE_TIME(true);
    SET_NEEDS_PRE_SPIKE_EVENT_TIME(true);
    
    SET_NEEDS_PREV_PRE_SPIKE_TIME(true);  
    SET_NEEDS_PREV_POST_SPIKE_TIME(true);
    SET_NEEDS_PREV_PRE_SPIKE_EVENT_TIME(true);
};

IMPLEMENT_MODEL(STDPDopamine);

