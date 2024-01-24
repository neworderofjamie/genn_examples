// GeNN includes
#include "modelSpec.h"

// Model includes
#include "parameters.h"

class DVS : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(DVS);
    SET_THRESHOLD_CONDITION_CODE("spikeVector[id / 32] & (1 << (id % 32))");
    SET_EXTRA_GLOBAL_PARAMS( {{"spikeVector", "uint32_t*"}} );
};
IMPLEMENT_SNIPPET(DVS);

void modelDefinition(ModelSpec &model)
{
    model.setDT(Parameters::timestep);
    model.setName("optical_flow");

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters for P population
    ParamValues lifParams{
        {"C", 1.0},
        {"TauM", 20.0},
        {"Vrest", -60.0},
        {"Vreset", -60.0},
        {"Vthresh", -50.0},
        {"Ioffset", 0.0},
        {"TauRefrac", 1.0}};

    // LIF initial conditions
    VarValues lifInit{
        {"V", -60.0},
        {"RefracTime", 0.0}};

    ParamValues dvsMacroPixelWeightUpdateInit{
        {"g", 0.8}};

    ParamValues macroPixelOutputExcitatoryWeightUpdateInit{
        {"g", 1.0}};

    ParamValues macroPixelOutputInhibitoryWeightUpdateInit{
        {"g", -0.5}};

    // Exponential current parameters
    ParamValues macroPixelPostSynParams{
        {"tau", 5.0}};

    ParamValues outputExcitatoryPostSynParams{
        {"tau", 25.0}};

    ParamValues outputInhibitoryPostSynParams{
        {"tau", 50.0}};

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    // Create IF_curr neuron
    auto *dvs = model.addNeuronPopulation<DVS>("DVS", Parameters::inputSize * Parameters::inputSize);
    auto *macroPixel = model.addNeuronPopulation<NeuronModels::LIF>("MacroPixel", Parameters::macroPixelSize * Parameters::macroPixelSize,
                                                                    lifParams, lifInit);

    auto *output = model.addNeuronPopulation<NeuronModels::LIF>("Output", Parameters::detectorSize * Parameters::detectorSize * Parameters::DetectorMax,
                                                                lifParams, lifInit);

    //------------------------------------------------------------------------
    // Synapse populations
    //------------------------------------------------------------------------
    auto *dvsMacroPixel = model.addSynapsePopulation(
        "DVS_MacroPixel", SynapseMatrixType::SPARSE,
        dvs, macroPixel,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(dvsMacroPixelWeightUpdateInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(macroPixelPostSynParams));

    auto *macroPixelOutputExcitatory = model.addSynapsePopulation(
        "MacroPixel_Output_Excitatory", SynapseMatrixType::SPARSE,
        macroPixel, output,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(macroPixelOutputExcitatoryWeightUpdateInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(outputExcitatoryPostSynParams));

    auto *macroPixelOutputInhibitory = model.addSynapsePopulation(
        "MacroPixel_Output_Inhibitory", SynapseMatrixType::SPARSE,
        macroPixel, output,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(macroPixelOutputInhibitoryWeightUpdateInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(outputInhibitoryPostSynParams));
    
    dvsMacroPixel->setMaxConnections(1);
    macroPixelOutputExcitatory->setMaxConnections(Parameters::DetectorMax);
    macroPixelOutputInhibitory->setMaxConnections(Parameters::DetectorMax);
    // Use zero-copy for input and output spikes as we want to access them every timestep
    //dvs->setSpikeZeroCopyEnabled(true);
    //output->setSpikeZeroCopyEnabled(true);
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
}
