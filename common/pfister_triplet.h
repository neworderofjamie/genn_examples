#pragma once

// GeNN includes
#include "modelSpec.h"

//----------------------------------------------------------------------------
// PfisterTriplet
//----------------------------------------------------------------------------
class PfisterTriplet : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(PfisterTriplet, 10, 1, 2, 2);

    SET_PARAM_NAMES({
      "tauPlus",  // 0 - Potentiation time constant (ms)
      "tauMinus", // 1 - Depression time constant (ms)
      "tauX",     // 2 - Potentiation time constant (ms)
      "tauY",     // 3 - Depression time constant (ms)
      "A2Plus",    // 4 - Rate of potentiation
      "A2Minus",    // 5 - Rate of potentiation
      "A3Plus",    // 6 - Rate of potentiation
      "A3Minus",    // 7 - Rate of potentiation
      "Wmin",     // 8 - Minimum weight
      "Wmax",     // 9 - Maximum weight
    });

    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"r1", "scalar"}, {"r2", "scalar"}});
    SET_POST_VARS({{"o1", "scalar"}, {"o2", "scalar"}});

    SET_SIM_PREAMBLE_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "$(r1) = ($(r1) * exp(-dt / $(tauPlus))) + 1.0;\n"
        "$(r2) =  ($(sT_pre) == 0.0) ? 0.0 : ($(r2) + 1.0) * exp(-dt / $(tauX));\n");

    SET_SIM_CODE(
        "$(addtoinSyn) = $(g);\n"
        "$(updatelinsyn);\n"
        "scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar o1 = $(o1) * exp(-dt / $(tauMinus));\n"
        "    scalar newWeight = $(g) - o1 * ($(A2Minus) + ($(A3Minus) * $(r2)));\n"
        "    $(g) = (newWeight < $(Wmin)) ? $(Wmin) : newWeight;\n"
        "}\n");

    SET_LEARN_POST_PREAMBLE_CODE(
        "scalar dt = $(t) - $(sT_post);\n"
        "$(o1) = ($(o1) * exp(-dt / $(tauPlus))) + 1.0;\n"
        "$(o2) = ($(sT_post) == 0.0) ? 0.0 : ($(o2) + 1.0) * exp(-dt / $(tauX));\n");

    SET_LEARN_POST_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0)\n"
        "{\n"
        "    scalar r1 = $(r1) * exp(-dt / $(tauPlus));\n"
        "    scalar newWeight = $(g) + r1 * ($(A2Plus) + ($(A3Plus) * $(o2)));\n"
        "    $(g) = (newWeight > $(Wmax)) ? $(Wmax) : newWeight;\n"
        "}\n");

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};

IMPLEMENT_MODEL(PfisterTriplet);