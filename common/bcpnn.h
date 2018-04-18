#pragma once

// GeNN includes
#include "modelSpec.h"

//----------------------------------------------------------------------------
// BCPNNTwoTrace
//----------------------------------------------------------------------------
class BCPNNTwoTrace : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(BCPNNTwoTrace, 7, 3, 2, 2);

    SET_PARAM_NAMES({
      "tauZi",                  // 0 - Time constant of presynaptic primary trace (ms)
      "tauZj",                  // 1 - Time constant of postsynaptic primary trace (ms)
      "tauP",                   // 2 - Time constant of probability trace
      "fMax",                   // 3 - Maximum firing frequency (Hz)
      "wMax",                   // 4 - Maximum weight
      "weightEnabled",          // 5 - Should weights get applied to synapses
      "plasticityEnabled"});    // 6 - Should weights be updated

    SET_VARS({{"g", "scalar"}, {"PijStar", "scalar"}, {"lastUpdateTime", "scalar"}});
    SET_PRE_VARS({{"ZiStar", "scalar"}, {"PiStar", "scalar"}});
    SET_POST_VARS({{"ZjStar", "scalar"}, {"PjStar", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Ai", [](const vector<double> &pars, double){ return 1000.0 / (pars[3] * (pars[0] - pars[2])); }},
        {"Aj", [](const vector<double> &pars, double){ return 1000.0 / (pars[3] * (pars[1] - pars[2])); }},
        {"Aij", [](const vector<double> &pars, double){ return (1000000.0 / (pars[0] + pars[1])) / ((pars[3]  * pars[3]) * ((1.0 / ((1.0 / pars[0]) + (1.0 / pars[1]))) - pars[2])); }},
        {"Epsilon", [](const vector<double> &pars, double){ return 1000.0 / (pars[3] * pars[2]); }}});

    SET_PRE_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_pre);\n"
        "$(ZiStar) = ($(ZiStar) * exp(-dt / $(tauZi))) + 1.0;\n"
        "$(PiStar) = ($(PiStar) * exp(-dt / $(tauP))) + 1.0;\n");

    SET_POST_SPIKE_CODE(
        "scalar dt = $(t) - $(sT_post);\n"
        "$(ZjStar) = ($(ZjStar) * exp(-dt / $(tauZj))) + 1.0;\n"
        "$(PjStar) = ($(PjStar) * exp(-dt / $(tauP))) + 1.0;\n");

    SET_SIM_CODE(
        "if($(weightEnabled)) {\n"
        "   $(addtoinSyn) = $(g);\n"
        "   $(updatelinsyn);\n"
        "}\n"
        "if($(plasticityEnabled)) {\n"
        "   scalar timeSinceLastUpdate = $(t) - $(lastUpdateTime);\n"
        "   scalar timeSinceLastPost = $(t) - $(sT_post);\n"
        "   scalar newZjStar = $(ZjStar) * exp(-timeSinceLastPost / $(tauZj));\n"
        "   $(PijStar) = ($(PijStar) * exp(-timeSinceLastUpdate / $(tauP))) + newZjStar;\n"
        "   scalar Pi = $(Ai) * ($(ZiStar) - $(PiStar));\n"
        "   scalar Pj = $(Aj) * (newZjStar - $(PjStar));\n"
        "   scalar Pij = $(Aij) * (($(ZiStar) * newZjStar) - $(PijStar));\n"
        "   scalar logPij = log(Pij + ($(Epsilon) * $(Epsilon)));\n"
        "   scalar logPiPj = log((Pi + $(Epsilon)) * (Pj + $(Epsilon)));\n"
        "   $(g) = logPij - logPiPj;\n"
        "   $(lastUpdateTime) = $(t);\n"
        "}\n");

    SET_LEARN_POST_CODE(
        "if($(plasticityEnabled)) {\n"
        "   scalar timeSinceLastUpdate = $(t) - $(lastUpdateTime);\n"
        "   scalar timeSinceLastPre = $(t) - $(sT_pre);\n"
        "   scalar newZiStar = $(ZiStar) * exp(-timeSinceLastPre / $(tauZi));\n"
        "   $(PijStar) = ($(PijStar) * exp(-timeSinceLastUpdate / $(tauP))) + newZiStar;\n"
        "   scalar Pi = $(Ai) * (newZiStar - $(PiStar));\n"
        "   scalar Pj = $(Aj) * ($(ZjStar) - $(PjStar));\n"
        "   scalar Pij = $(Aij) * ((newZiStar * $(ZjStar)) - $(PijStar));\n"
        "   scalar logPij = log(Pij + ($(Epsilon) * $(Epsilon)));\n"
        "   scalar logPiPj = log((Pi + $(Epsilon)) * (Pj + $(Epsilon)));\n"
        "   $(g) = logPij - logPiPj;\n"
        "   $(lastUpdateTime) = $(t);\n"
        "}\n");

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
};

IMPLEMENT_MODEL(BCPNNTwoTrace);