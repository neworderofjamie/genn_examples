#include "modelSpec.h"

#include "../common/eprop_models.h"

#include "models.h"
#include "parameters.h"

//----------------------------------------------------------------------------
// OutputClassification
//----------------------------------------------------------------------------
class OutputClassification : public NeuronModels::Base
{
public:
    DECLARE_MODEL(OutputClassification, 1, 5);

    SET_PARAM_NAMES({"TauOut"});    // Membrane time constant [ms]

    SET_VARS({{"Y", "scalar"}, {"Pi", "scalar"}, {"E", "scalar"},
              {"B", "scalar"}, {"DeltaB", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Kappa", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});

    SET_SIM_CODE(
        "const int globalTimestep = (int)$(t);\n"
        "const int trial = globalTimestep / ((28 * 28 * 2) + 20);\n"
        "const int timestep = globalTimestep % ((28 * 28 * 2) + 20);\n"
        "$(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);\n"
        "scalar m = $(Y);\n"
        "m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x1));\n"
        "m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x2));\n"
        "m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x4));\n"
        "m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x8));\n"
        "const scalar expPi = exp($(Y) - m);\n"
        "scalar sumExpPi = expPi;\n"
        "sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x1);\n"
        "sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x2);\n"
        "sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x4);\n"
        "sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x8);\n"
        "$(Pi) = expPi / sumExpPi;\n"
        "if(timestep < (28 * 28 * 2)) {\n"
        "   $(E) = 0.0;\n"
        "}\n"
        "else {\n"
        "   const scalar piStar = ($(id) == $(labels)[$(indices)[trial]]) ? 1.0 : 0.0;\n"
        "   $(E) = $(Pi) - piStar;\n"
        "}\n"
        "$(DeltaB) += $(E);\n");
    
    SET_EXTRA_GLOBAL_PARAMS({{"indices", "unsigned int*"}, {"labels", "uint8_t*"}});
    
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(OutputClassification);

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

    model.setDT(Parameters::timestepMs);
    model.setName("s_mnist");
    model.setMergePostsynapticModels(true);
    model.setTiming(Parameters::timingEnabled);
    model.setSeed(1234);

    //---------------------------------------------------------------------------
    // Parameters and state variables
    //---------------------------------------------------------------------------
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

    // Output population
    OutputClassification::ParamValues outputParamVals(20.0);    // Membrane time constant [ms]

    OutputClassification::VarValues outputInitVals(
        0.0,    // Y
        0.0,    // Pi
        0.0,    // E
        0.0,    // B
        0.0);   // DeltaB

    EPropALIF::ParamValues epropALIFParamVals(
        20.0,                                           // Eligibility trace time constant [ms]
        2000.0,                                         // Neuron adaption time constant [ms]
        1.0 / ((double)Parameters::batchSize * 1000.0), // Regularizer strength
        10.0,                                           // Target spike rate [Hz]
        500.0,                                          // Firing rate averaging time constant [ms]
        0.0174);                                        // Scale of neuron adaption [mV]

    EProp::PreVarValues epropPreInitVals(
        0.0);   // ZFilter

    EProp::PostVarValues epropPostInitVals(
        0.0,    // Psi
        0.0);   // FAvg

    // Feedforward input->recurrent connections
    InitVarSnippet::Normal::ParamValues inputRecurrentWeightDist(0.0, weight0 / sqrt(Parameters::numInputNeurons));
    EPropALIF::VarValues inputRecurrentALIFInitVals(
#ifdef RESUME_EPOCH
        uninitialisedVar(),                                         // g
#else
        initVar<InitVarSnippet::Normal>(inputRecurrentWeightDist),  // g
#endif
        0.0,                                                        // eFiltered
        0.0,                                                        // epsilonA
        0.0);                                                       // DeltaG

    // Recurrent connections
    InitVarSnippet::Normal::ParamValues recurrentRecurrentWeightDist(0.0, weight0 / sqrt(Parameters::numRecurrentNeurons * 2));
    EPropALIF::VarValues recurrentRecurrentALIFInitVals(
#ifdef RESUME_EPOCH
        uninitialisedVar(),                                             // g
#else
        initVar<InitVarSnippet::Normal>(recurrentRecurrentWeightDist),  // g
#endif
        0.0,                                                            // eFiltered
        0.0,                                                            // epsilonA
        0.0);                                                           // DeltaG
        
    // Feedforward recurrent->output connections
    OutputLearning::ParamValues recurrentOutputParamVals(
        20.0);   // Eligibility trace time constant [ms]

    OutputLearning::PreVarValues recurrentOutputPreInitVals(
        0.0);   // ZFilter

    InitVarSnippet::Normal::ParamValues recurrentOutputWeightDist(0.0, weight0 / sqrt(Parameters::numRecurrentNeurons * 2));
    OutputLearning::VarValues recurrentOutputInitVals(
#ifdef RESUME_EPOCH
        uninitialisedVar(),                                             // g
#else
        initVar<InitVarSnippet::Normal>(recurrentOutputWeightDist), // g
#endif
        0.0);                                                       // DeltaG

    // Feedback connections
    // **HACK** this is actually a nasty corner case for the initialisation rules
    // We really want this uninitialised as we are going to copy over transpose 
    // But then initialiseSparse would copy over host values
    Continuous::VarValues outputRecurrentInitVals(0.0);  // g

    //---------------------------------------------------------------------------
    // Neuron populations
    //---------------------------------------------------------------------------
    auto *input = model.addNeuronPopulation<InputSequential>("Input", Parameters::numInputNeurons,
                                                             {}, {});

    auto *recurrentALIF = model.addNeuronPopulation<RecurrentALIF>("RecurrentALIF", Parameters::numRecurrentNeurons,
                                                                   recurrentALIFParamVals, recurrentALIFInitVals);

    auto *output = model.addNeuronPopulation<OutputClassification>("Output", Parameters::numOutputNeurons,
                                                                   outputParamVals, outputInitVals);

#ifdef ENABLE_RECORDING
    input->setSpikeRecordingEnabled(true);
    recurrentALIF->setSpikeRecordingEnabled(true);
#endif

    //---------------------------------------------------------------------------
    // Synapse populations
    //---------------------------------------------------------------------------
    // Input->recurrent connections
    auto *inputRecurrent = model.addSynapsePopulation<EPropALIF, PostsynapticModels::DeltaCurr>(
        "InputRecurrentALIF", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Input", "RecurrentALIF",
        epropALIFParamVals, inputRecurrentALIFInitVals, epropPreInitVals, epropPostInitVals,
        {}, {});

    // Recurrent->recurrent connections
    auto *recurrentRecurrent = model.addSynapsePopulation<EPropALIF, PostsynapticModels::DeltaCurr>(
        "ALIFALIFRecurrent", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "RecurrentALIF", "RecurrentALIF",
        epropALIFParamVals, recurrentRecurrentALIFInitVals, epropPreInitVals, epropPostInitVals,
        {}, {});

    // Recurrent->output connections
    auto *recurrentOutput = model.addSynapsePopulation<OutputLearning, PostsynapticModels::DeltaCurr>(
        "RecurrentALIFOutput", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "RecurrentALIF", "Output",
        recurrentOutputParamVals, recurrentOutputInitVals, recurrentOutputPreInitVals, {},
        {}, {});

    // Output->recurrent populations
    auto *outputRecurrent = model.addSynapsePopulation<Continuous, Feedback>(
        "OutputRecurrentALIF", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Output", "RecurrentALIF",
        {}, outputRecurrentInitVals,
        {}, {});
    
    //---------------------------------------------------------------------------
    // Custom updates
    //---------------------------------------------------------------------------
    AdamOptimizer::ParamValues adamParams(Parameters::adamBeta1, Parameters::adamBeta2, 1E-8);
    AdamOptimizer::VarValues adamVarValues(0.0, 0.0);
    
    AdamOptimizer::VarReferences adamBiasVarReferences(createVarRef(output, "DeltaB"),    // Gradient 
                                                       createVarRef(output, "B"));        // Variable
    model.addCustomUpdate<AdamOptimizer>("OutputBiasOptimiser", "GradientLearn",
                                         adamParams, adamVarValues, adamBiasVarReferences);

    AdamOptimizer::WUVarReferences adamInputRecurrentVarReferences(
        createWUVarRef(inputRecurrent, "DeltaG"),    // Gradient 
        createWUVarRef(inputRecurrent, "g"));        // Variable
    model.addCustomUpdate<AdamOptimizer>("InputRecurrentWeightOptimiser", "GradientLearn",
                                         adamParams, adamVarValues, adamInputRecurrentVarReferences);

    AdamOptimizer::WUVarReferences adamRecurrentRecurrentVarReferences(
        createWUVarRef(recurrentRecurrent, "DeltaG"),    // Gradient 
        createWUVarRef(recurrentRecurrent, "g"));        // Variable
    model.addCustomUpdate<AdamOptimizer>("RecurrentRecurrentWeightOptimiser", "GradientLearn",
                                         adamParams, adamVarValues, adamRecurrentRecurrentVarReferences);     

    AdamOptimizer::WUVarReferences adamRecurrentOutputVarReferences(
        createWUVarRef(recurrentOutput, "DeltaG"),                      // Gradient 
        createWUVarRef(recurrentOutput, "g", outputRecurrent, "g"));    // Variable
    model.addCustomUpdate<AdamOptimizer>("RecurrentOutputWeightOptimiser", "GradientLearn",
                                         adamParams, adamVarValues, adamRecurrentOutputVarReferences);

    CustomUpdateModels::Transpose::WUVarReferences transposeRecurrentOutputVarReferences(
        createWUVarRef(recurrentOutput, "g", outputRecurrent, "g"));    // Variable
    model.addCustomUpdate<CustomUpdateModels::Transpose>("RecurrentOutputWeightTranspose", "CalculateTranspose",
                                                         {}, {}, transposeRecurrentOutputVarReferences);
}
