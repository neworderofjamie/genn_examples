#pragma once

// GeNN includes
#include "modelSpec.h"

//----------------------------------------------------------------------------
// OpenCVLIF
//----------------------------------------------------------------------------
//! Leaky integrate-and-fire neuron solved algebraically
//! Input current comes from a external array of floats and 
//! a stride  (as provided by an OpenCV PtrStep struct)
class OpenCVLIF : public NeuronModels::Base
{
public:
    DECLARE_MODEL(OpenCVLIF, 8, 2);

    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0)\n"
        "{\n"
        "  unsigned int x = $(id) % (unsigned int)($(Resolution));\n"
        "  unsigned int y = $(id) / (unsigned int)($(Resolution));\n"
        "  unsigned int index = (y * $(step)) + x;\n"
        "  scalar inputCurrent = *($(inputCurrents) + index);\n"
        "  scalar alpha = ((inputCurrent + $(Ioffset)) * $(Rmembrane)) + $(Vrest);\n"
        "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else\n"
        "{\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "Ioffset",    // Offset current
        "TauRefrac",  // Refractory time [ms]
        "Resolution", // Resolution of population
    });

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});
    
    SET_EXTRA_GLOBAL_PARAMS({{"inputCurrents", "float*"}, {"step", "unsigned int"}});
};
IMPLEMENT_MODEL(OpenCVLIF);