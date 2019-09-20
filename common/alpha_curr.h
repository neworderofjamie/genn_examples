#pragma once

// GeNN includes
#include "modelSpec.h"

//----------------------------------------------------------------------------
// AlphaCurr
//----------------------------------------------------------------------------
//! Current-based synapse model with alpha shaping
class AlphaCurr : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(AlphaCurr, 1, 1);

    SET_DECAY_CODE(
        "$(x) = (DT * $(expDecay) * $(inSyn) * $(init)) + ($(expDecay) * $(x));\n"
        "$(inSyn)*=$(expDecay);\n");

    SET_CURRENT_CONVERTER_CODE("$(x)");

    SET_PARAM_NAMES({"tau"});
    
    SET_VARS({{"x", "scalar"}});

    SET_DERIVED_PARAMS({
        {"expDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"init", [](const std::vector<double> &pars, double){ return (std::exp(1) / pars[0]); }}});
};
IMPLEMENT_MODEL(AlphaCurr);
