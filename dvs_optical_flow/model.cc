// GeNN includes
#include "modelSpec.h"

// Model includes
#include "parameters.h"


void modelDefinition(NNmodel &model)
{
    model.setDT(Parameters::timestep);
    model.setName("optical_flow");

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters for P population
    NeuronModels::LIF::ParamValues lifParams(
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        1.0);    // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        -60.0,        // 0 - V
        0.0);       // 1 - RefracTime

    WeightUpdateModels::StaticPulse::VarValues dvsMacroPixelWeightUpdateInit(
        0.8);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues macroPixelOutputExcitatoryWeightUpdateInit(
        1.0);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues macroPixelOutputInhibitoryWeightUpdateInit(
        -0.5);     // 0 - Wij (nA)

    // Exponential current parameters
    PostsynapticModels::ExpCurr::ParamValues macroPixelPostSynParams(
        5.0);         // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurr::ParamValues outputExcitatoryPostSynParams(
        25.0);         // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurr::ParamValues outputInhibitoryPostSynParams(
        50.0);         // 0 - TauSyn (ms)

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    // Create IF_curr neuron
    auto *dvs = model.addNeuronPopulation<NeuronModels::SpikeSource>("DVS", Parameters::inputSize * Parameters::inputSize,
                                                         {}, {});
    model.addNeuronPopulation<NeuronModels::LIF>("MacroPixel", Parameters::macroPixelSize * Parameters::macroPixelSize,
                                                 lifParams, lifInit);

    auto *output = model.addNeuronPopulation<NeuronModels::LIF>("Output", Parameters::detectorSize * Parameters::detectorSize * Parameters::DetectorMax,
                                                                lifParams, lifInit);

    //------------------------------------------------------------------------
    // Synapse populations
    //------------------------------------------------------------------------
    auto *dvsMacroPixel = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "DVS_MacroPixel", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "DVS", "MacroPixel",
        {}, dvsMacroPixelWeightUpdateInit,
        macroPixelPostSynParams, {});

    auto *macroPixelOutputExcitatory = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "MacroPixel_Output_Excitatory", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "MacroPixel", "Output",
        {}, macroPixelOutputExcitatoryWeightUpdateInit,
        outputExcitatoryPostSynParams, {});

    auto *macroPixelOutputInhibitory = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "MacroPixel_Output_Inhibitory", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "MacroPixel", "Output",
        {}, macroPixelOutputInhibitoryWeightUpdateInit,
        outputInhibitoryPostSynParams, {});
    
    dvsMacroPixel->setMaxConnections(1);
    macroPixelOutputExcitatory->setMaxConnections(Parameters::DetectorMax);
    macroPixelOutputInhibitory->setMaxConnections(Parameters::DetectorMax);
    // Use zero-copy for input and output spikes as we want to access them every timestep
    //dvs->setSpikeZeroCopyEnabled(true);
    //output->setSpikeZeroCopyEnabled(true);
}
