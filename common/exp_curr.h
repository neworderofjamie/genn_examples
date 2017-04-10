#pragma once

// GeNN includes
#include "modelSpec.h"

//----------------------------------------------------------------------------
// ExpCurr
//----------------------------------------------------------------------------
//! Current-based synapse model with direct current injection
class ExpCurr : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(ExpCurr, 1, 0);

    SET_DECAY_CODE("$(inSyn)*=$(expDecay);");

    SET_CURRENT_CONVERTER_CODE("$(inSyn)");

    SET_PARAM_NAMES({"tau"});

    SET_DERIVED_PARAMS({{"expDecay", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});
};
IMPLEMENT_MODEL(ExpCurr);