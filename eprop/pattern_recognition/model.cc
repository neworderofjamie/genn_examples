#include "modelSpec.h"

#include "../common/eprop_models.h"

#include "parameters.h"

constexpr double PI = 3.14159265358979323846264338327950288419;

//----------------------------------------------------------------------------
// Input
//----------------------------------------------------------------------------
class Input : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Input, 4, 1);

    SET_PARAM_NAMES({
        "GroupSize",        // Number of neurons in each group
        "ActiveInterval",   // How long each group is active for [ms]
        "ActiveRate",       // Rate active neurons fire at [Hz]
        "PatternLength"});  // Pattern length [ms]

    SET_VARS({{"RefracTime", "scalar"}});

    SET_DERIVED_PARAMS({
        {"TauRefrac", [](const std::vector<double> &pars, double){ return 1000.0 / pars[3]; }}});

    SET_SIM_CODE(
        "const scalar tPattern = fmod($(t), $(PatternLength));\n"
        "const unsigned int neuronGroup = $(id) / (unsigned int)$(GroupSize);\n"
        "const scalar groupStartTime = neuronGroup * $(ActiveInterval);\n"
        "const scalar groupEndTime = groupStartTime + $(ActiveInterval);\n"
        "if ($(RefracTime) > 0.0) {\n"
        "  $(RefracTime) -= DT;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE(
        "tPattern > groupStartTime && tPattern < groupEndTime && $(RefracTime) <= 0.0");

    SET_RESET_CODE("$(RefracTime) = $(TauRefrac);\n");

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Input);

//----------------------------------------------------------------------------
// Output
//----------------------------------------------------------------------------
class OutputRegression : public NeuronModels::Base
{
public:
    DECLARE_MODEL(OutputRegression, 6, 9);

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
IMPLEMENT_MODEL(OutputRegression);

void modelDefinition(ModelSpec &model)
{
    // Calculate weight scaling factor
    // **NOTE** "Long short-term memory and 
    // learning-to-learn in networks of spiking neurons"
    // suggests that this should be (1 Volt * DT)/Rm but
    // that results in 1E-9 or something which is never 
    // going to make these neurons spike - the paper then 
    // SORT OF suggests that they use 1.0
    const double weight0 = 1.0;

    model.setDT(1.0);
    model.setName("pattern_recognition_1_1");
    model.setMergePostsynapticModels(true);
    model.setTiming(Parameters::timingEnabled);

    //---------------------------------------------------------------------------
    // Parameters and state variables
    //---------------------------------------------------------------------------
    // Input population
    Input::ParamValues inputParamVals(
        4,          // Number of neurons in each group
        200.0,      // How long each group is active for [ms]
        100.0,      // Rate active neurons fire at [Hz]
        1000.0);    // Pattern length [ms]

    Input::VarValues inputInitVals(
        0.0);   // Refrac time

    // Recurrent population
    Recurrent::ParamValues recurrentParamVals(
        20.0,   // Membrane time constant [ms]
        0.61,   // Spiking threshold [mV]
        5.0);   // Refractory time constant [ms]

    Recurrent::VarValues recurrentInitVals(
        0.0,    // V
        0.0,    // RefracTime
        0.0);   // E

    // Output population
    OutputRegression::ParamValues outputParamVals(
        20.0,       // Membrane time constant [ms]
        0.0,        // Bias [mV]
        2.0,        // Frequency of sine wave 1 [Hz]
        3.0,        // Frequency of sine wave 2 [Hz]
        5.0,        // Frequency of sine wave 3 [Hz]
        1000.0);    // Pattern length [ms]

    InitVarSnippet::Uniform::ParamValues outputAmplDist(0.5, 2.0);
    InitVarSnippet::Uniform::ParamValues outputPhaseDist(0.0, 2.0 * PI);
    OutputRegression::VarValues outputInitVals(
        0.0,                                                // Y
        0.0,                                                // Y*
        0.0,                                                // E
        initVar<InitVarSnippet::Uniform>(outputAmplDist),   // Ampl1
        initVar<InitVarSnippet::Uniform>(outputAmplDist),   // Ampl2
        initVar<InitVarSnippet::Uniform>(outputAmplDist),   // Ampl3
        initVar<InitVarSnippet::Uniform>(outputPhaseDist),  // Phase1
        initVar<InitVarSnippet::Uniform>(outputPhaseDist),  // Phase2
        initVar<InitVarSnippet::Uniform>(outputPhaseDist)); // Phase3

    EProp::ParamValues epropParamVals(
        20.0,       // Eligibility trace time constant [ms]
        3.0,        // Regularizer strength
        10.0,       // Target spike rate [Hz]
        500.0);     // Firing rate averaging time constant [ms]
    
    EProp::PreVarValues epropPreInitVals(
        0.0);   // ZFilter

    EProp::PostVarValues epropPostInitVals(
        0.0,    // Psi
        0.0);   // FAvg

    // Feedforward input->recurrent connections
    InitVarSnippet::Normal::ParamValues inputRecurrentWeightDist(0.0, weight0 / sqrt(Parameters::numInputNeurons));
    EProp::VarValues inputRecurrentInitVals(
        initVar<InitVarSnippet::Normal>(inputRecurrentWeightDist),  // g
        0.0,                                                        // eFiltered
        0.0,                                                        // DeltaG
        0.0,                                                        // M
        0.0);                                                       // V

    // Recurrent connections
    InitVarSnippet::Normal::ParamValues recurrentRecurrentWeightDist(0.0, weight0 / sqrt(Parameters::numRecurrentNeurons));
    EProp::VarValues recurrentRecurrentInitVals(
        initVar<InitVarSnippet::Normal>(recurrentRecurrentWeightDist),  // g
        0.0,                                                            // eFiltered
        0.0,                                                            // DeltaG
        0.0,                                                            // M
        0.0);                                                           // V

    // Feedforward recurrent->output connections
    OutputLearning::ParamValues recurrentOutputParamVals(
        20.0);   // Eligibility trace time constant [ms]

    OutputLearning::PreVarValues recurrentOutputPreInitVals(
        0.0);   // ZFilter

    InitVarSnippet::Normal::ParamValues recurrentOutputWeightDist(0.0, weight0 / sqrt(Parameters::numRecurrentNeurons * Parameters::deepRRecurrentConnectivity));
    OutputLearning::VarValues recurrentOutputInitVals(
        initVar<InitVarSnippet::Normal>(recurrentOutputWeightDist), // g
        0.0,                                                        // DeltaG
        0.0,                                                        // M
        0.0);                                                       // V

    // Feedback connections
    // **HACK** this is actually a nasty corner case for the initialisation rules
    // We really want this uninitialised as we are going to copy over transpose 
    // But then initialiseSparse would copy over host values
    Continuous::VarValues outputRecurrentInitVals(0.0);

    //---------------------------------------------------------------------------
    // Neuron populations
    //---------------------------------------------------------------------------
    auto *input = model.addNeuronPopulation<Input>("Input", Parameters::numInputNeurons,
                                                   inputParamVals, inputInitVals);
    auto *recurrent = model.addNeuronPopulation<Recurrent>("Recurrent", Parameters::numRecurrentNeurons,
                                                           recurrentParamVals, recurrentInitVals);
    model.addNeuronPopulation<OutputRegression>("Output", Parameters::numOutputNeurons,
                                                outputParamVals, outputInitVals);
    
    input->setSpikeRecordingEnabled(true);
    recurrent->setSpikeRecordingEnabled(true);

    //---------------------------------------------------------------------------
    // Synapse populations
    //---------------------------------------------------------------------------
    model.addSynapsePopulation<EProp, PostsynapticModels::DeltaCurr>(
        "InputRecurrent", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Input", "Recurrent",
        epropParamVals, inputRecurrentInitVals, epropPreInitVals, epropPostInitVals,
        {}, {});

    model.addSynapsePopulation<Continuous, Feedback>(
        "OutputRecurrent", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Output", "Recurrent",
        {}, outputRecurrentInitVals,
        {}, {});

#ifdef USE_DEEP_R
    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(Parameters::deepRRecurrentConnectivity); // 0 - prob
    model.addSynapsePopulation<EProp, PostsynapticModels::DeltaCurr>(
        "RecurrentRecurrent", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "Recurrent", "Recurrent",
        epropParamVals, recurrentRecurrentInitVals, epropPreInitVals, epropPostInitVals,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
#else
    model.addSynapsePopulation<EProp, PostsynapticModels::DeltaCurr>(
        "RecurrentRecurrent", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Recurrent", "Recurrent",
        epropParamVals, recurrentRecurrentInitVals, epropPreInitVals, epropPostInitVals,
        {}, {});
#endif
    
    model.addSynapsePopulation<OutputLearning, PostsynapticModels::DeltaCurr>(
        "RecurrentOutput", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Recurrent", "Output",
        recurrentOutputParamVals, recurrentOutputInitVals, recurrentOutputPreInitVals, {},
        {}, {});
}
