#include "modelSpec.h"

#include "../common/eprop_models.h"

#include "parameters.h"

//----------------------------------------------------------------------------
// Input
//----------------------------------------------------------------------------
class Input : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Input, 0, 1);

    SET_THRESHOLD_CONDITION_CODE("$(gennrand_uniform) >= exp(-($(rate) / 1000.0) * DT)");

    SET_VARS({{"rate", "scalar"}});
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Input);

//----------------------------------------------------------------------------
// OutputClassification
//----------------------------------------------------------------------------
/*class OutputClassification : public NeuronModels::Base
{
public:
    DECLARE_MODEL(OutputClassification, 6, 9);

    SET_PARAM_NAMES({
        "TauOut",           // Membrane time constant [ms]
        "Bias",             // Bias [mV]
        "Freq1",            // Frequency of sine wave 1 [Hz]
        "Freq2",            // Frequency of sine wave 2 [Hz]
        "Freq3",            // Frequency of sine wave 3 [Hz]
        "PatternLength"});  // Pattern length [ms]

    SET_VARS({{"Y", "scalar"}, {"YStar", "scalar"}, {"E", "scalar"},
              {"Ampl1", "scalar", VarAccess::READ_ONLY}, {"Ampl2", "scalar", VarAccess::READ_ONLY}, {"Ampl3", "scalar", VarAccess::READ_ONLY},
              {"Phase1", "scalar", VarAccess::READ_ONLY}, {"Phase2", "scalar", VarAccess::READ_ONLY}, {"Phase3", "scalar", VarAccess::READ_ONLY}});

    SET_DERIVED_PARAMS({
        {"Kappa", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"Freq1Radians", [](const std::vector<double> &pars, double){ return pars[2] * 2.0 * PI / 1000.0; }},
        {"Freq2Radians", [](const std::vector<double> &pars, double){ return pars[3] * 2.0 * PI / 1000.0; }},
        {"Freq3Radians", [](const std::vector<double> &pars, double){ return pars[4] * 2.0 * PI / 1000.0; }}});

    SET_SIM_CODE(
        "$(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(Bias);\n"
        "const scalar tPattern = fmod($(t), $(PatternLength));\n"
        "$(YStar) = $(Ampl1) * sin(($(Freq1Radians) * tPattern) + $(Phase1));\n"
        "$(YStar) += $(Ampl2) * sin(($(Freq2Radians) * tPattern) + $(Phase2));\n"
        "$(YStar) += $(Ampl3) * sin(($(Freq3Radians) * tPattern) + $(Phase3));\n"
        "$(E) = $(Y) - $(YStar);\n");

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(OutputClassification);*/

void modelDefinition(ModelSpec &model)
{
    model.setDT(Parameters::timestepMs);
    model.setName("evidence_accumulation");
    model.setMergePostsynapticModels(true);
    model.setTiming(Parameters::timingEnabled);

    //---------------------------------------------------------------------------
    // Parameters and state variables
    //---------------------------------------------------------------------------
    // Input population
    Input::VarValues inputInitVals(
        Parameters::inactiveRateHz);    // rate

    // Recurrent LIF population
    Recurrent::ParamValues recurrentParamVals(
        20.0,   // Membrane time constant [ms]
        0.6,   // Spiking threshold [mV]
        5.0);   // Refractory time constant [ms]

    Recurrent::VarValues recurrentInitVals(
        0.0,    // V
        0.0,    // RefracTime
        0.0);   // E

    // Recurrent ALIF population
    RecurrentALIF::ParamValues recurrentALIFParamVals(
        20.0,       // Membrane time constant [ms]
        2000.0,     // Adaption time constant [ms]
        0.6,        // Spiking threshold [mV]
        5.0,        // Refractory time constant [ms]
        0.0174);    // Scale of adaption [mV]

    RecurrentALIF::VarValues recurrentALIFInitVals(
        0.0,    // V
        0.0,    // A
        0.0,    // RefracTime
        0.0);   // E

    //---------------------------------------------------------------------------
    // Neuron populations
    //---------------------------------------------------------------------------
    model.addNeuronPopulation<Input>("Input", Parameters::numInputNeurons,
                                     {}, inputInitVals);
}