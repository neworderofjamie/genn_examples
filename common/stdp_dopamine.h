#pragma once

// GeNN includes
#include "modelSpec.h"

//----------------------------------------------------------------------------
// STDPDopamine
//----------------------------------------------------------------------------
class STDPDopamine : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(STDPDopamine, 8, 3);

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
        {"c", "scalar"},    // Synaptic tag
        {"tC", "scalar"},   // Time of last synaptic tag update **YUCK** if sT_post and sT_pre were better defined this would be max(sT_pre, sT_post, $(tD))
    });

    SET_SIM_CODE(
        "$(addtoinSyn) = $(g);\n"
        "$(updatelinsyn);\n"
        "// Calculate how much tag has decayed since last update\n"
        "scalar tagDT = $(t) - $(tC);\n"
        "scalar tagDecay = exp(-tagDT / $(tauC));\n"
        "// Calculate how much dopamine has decayed since last update\n"
        "scalar dopamineDT = $(t) - $(tD);\n"
        "scalar dopamineDecay = exp(-dopamineDT / $(tauD));\n"
        "// Calculate offset to integrate over correct area\n"
        "scalar offset = ($(tC) <= $(tD)) ? exp(-($(tD) - $(tC)) / $(tauC)) : exp(-($(tC) - $(tD)) / $(tauD));\n"
        "// Update weight\n"
        "$(g) += ($(c) * $(d) * $(scale)) * ((tagDecay * dopamineDecay) - offset);\n"
        "// Decay tag and apply STDP\n"
        "scalar newTag = $(c) * tagDecay;\n"
        "scalar dt = $(t) - $(sT_post);\n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar timing = exp(-dt / $(tauMinus));\n"
        "    newTag -= ($(aMinus) * timing);\n"
        "}\n"
        "// Write back updated tag and update time\n"
        "$(c) = newTag;\n"
        "$(tC) = $(t);\n");

    SET_EVENT_CODE(
        "// Calculate how much tag has decayed since last update\n"
        "scalar tagDT = $(t) - $(tC);\n"
        "scalar tagDecay = exp(-tagDT / $(tauC));\n"
        "// Calculate how much dopamine has decayed since last update\n"
        "scalar dopamineDT = $(t) - $(tD);\n"
        "scalar dopamineDecay = exp(-dopamineDT / $(tauD));\n"
        "// Calculate offset to integrate over correct area\n"
        "scalar offset = ($(tC) <= $(tD)) ? exp(-($(tD) - $(tC)) / $(tauC)) : exp(-($(tC) - $(tD)) / $(tauD));\n"
        "// Update weight\n"
        "$(g) += ($(c) * $(d) * $(scale)) * ((tagDecay * dopamineDecay) - offset);\n"
        "// Write back updated tag and update time\n"
        "$(c) *= tagDecay;\n"
        "$(tC) = $(t);\n");

    SET_LEARN_POST_CODE(
        "// Calculate how much tag has decayed since last update\n"
        "scalar tagDT = $(t) - $(tC);\n"
        "scalar tagDecay = exp(-tagDT / $(tauC));\n"
        "// Calculate how much dopamine has decayed since last update\n"
        "scalar dopamineDT = $(t) - $(tD);\n"
        "scalar dopamineDecay = exp(-dopamineDT / $(tauD));\n"
        "// Calculate offset to integrate over correct area\n"
        "scalar offset = ($(tC) <= $(tD)) ? exp(-($(tD) - $(tC)) / $(tauC)) : exp(-($(tC) - $(tD)) / $(tauD));\n"
        "// Update weight\n"
        "$(g) += ($(c) * $(d) * $(scale)) * ((tagDecay * dopamineDecay) - offset);\n"
        "// Decay tag and apply STDP\n"
        "scalar newTag = $(c) * tagDecay;\n"
        "scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar timing = exp(-dt / $(tauPlus));\n"
        "    newTag += ($(aPlus) * timing);\n"
        "}\n"
        "// Write back updated tag and update time\n"
        "$(c) = newTag;\n"
        "$(tC) = $(t);\n");

    SET_EVENT_THRESHOLD_CONDITION_CODE("$(injectDopamine)");

    SET_EXTRA_GLOBAL_PARAMS({
        {"injectDopamine", "bool"},
        {"tD", "scalar"},
        {"d", "scalar"}
    });

    SET_DERIVED_PARAMS({
        {"scale", [](const vector<double> &pars, double){ return 1.0 / -((1.0 / pars[2]) + (1.0 / pars[3])); }}
    });

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};

IMPLEMENT_MODEL(STDPDopamine);