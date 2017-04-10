#pragma once

// GeNN includes
#include "modelSpec.h"

//----------------------------------------------------------------------------
// STDPAdditive
//----------------------------------------------------------------------------
class STDPAdditive : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(STDPAdditive, 6, 1);

    SET_PARAM_NAMES({
      "tauPlus",  // 0 - Potentiation time constant (ms)
      "tauMinus", // 1 - Depression time constant (ms)
      "Aplus",    // 2 - Rate of potentiation
      "Aminus",   // 3 - Rate of depression
      "Wmin",     // 4 - Minimum weight
      "Wmax",     // 5 - Maximum weight
    });

    SET_VARS({{"g", "scalar"}});

    SET_SIM_CODE(
        "$(addtoinSyn) = $(g);\n"
        "$(updatelinsyn);\n"
        "scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar timing = exp(-dt / $(tauMinus));\n"
        "    scalar newWeight = $(g) - ($(Aminus) * timing);\n"
        "    $(g) = (newWeight < $(Wmin)) ? $(Wmin) : newWeight;\n"
        "}\n");
    SET_LEARN_POST_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar timing = exp(-dt / $(tauPlus));\n"
        "    scalar newWeight = $(g) + ($(Aplus) * timing);\n"
        "    $(g) = (newWeight > $(Wmax)) ? $(Wmax) : newWeight;\n"
        "}\n");

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};

IMPLEMENT_MODEL(STDPAdditive);