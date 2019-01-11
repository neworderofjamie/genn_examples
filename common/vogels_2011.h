#pragma once

// GeNN includes
#include "modelSpec.h"

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
        "$(addToInSyn, $(g));\n"
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