#pragma once

// GeNN includes
#include "currentSourceModels.h"

// Poisson current source with exponential shaping
class PoissonCurrentSourceExp : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(PoissonCurrentSourceExp, 3, 1);

    SET_INJECTION_CODE(
        "scalar p = 1.0f;\n"
        "unsigned int numSpikes = 0;\n"
        "do\n"
        "{\n"
        "    numSpikes++;\n"
        "    p *= $(gennrand_uniform);\n"
        "} while (p > $(expMinusLambda));\n"
        "$(iPoisson) += $(currentInit) * (scalar)(numSpikes - 1);\n"
        "$(injectCurrent, $(iPoisson));\n"
        "$(iPoisson) *= $(currentExpDecay);\n");

    SET_PARAM_NAMES({
        "rate",     // Poisson input rate [Hz]
        "weight",   // How much current each poisson spike adds [nA]
        "tauSyn"}); // Time constant of exponential shaping

    SET_DERIVED_PARAMS({
        {"expMinusLambda", [](const vector<double> &pars, double dt){ return std::exp(-(pars[0] / 1000.0) * dt); }},
        {"currentExpDecay", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[2]); }},
        {"currentInit", [](const vector<double> &pars, double dt){ return pars[1] * (1.0 - std::exp(-dt / pars[2])) * (pars[2] / dt); }}});

    SET_VARS({{"iPoisson", "scalar"}});
};
IMPLEMENT_MODEL(PoissonCurrentSourceExp);

// Poisson current source with alpha shaping
/*class PoissonCurrentSourceAlpha : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(PoissonCurrentSourceAlpha, 1, 0);

    SET_INJECTION_CODE("$(injectCurrent, ($(gennrand_uniform) * $(n) * 2.0) - $(n));\n");
    SET_PARAM_NAMES({"n"});
};
IMPLEMENT_MODEL(PoissonCurrentSourceAlpha);*/