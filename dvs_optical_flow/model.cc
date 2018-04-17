#include <cmath>
#include <vector>

// GeNN includes
#include "modelSpec.h"

// GeNN robotics includes
#include "exp_curr.h"
#include "lif.h"

// Model includes
#include "parameters.h"


void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(Parameters::timestep);
    model.setName("optical_flow");

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters for P population
    LIF::ParamValues lifParams(
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        1.0);    // 6 - TauRefrac

    // LIF initial conditions
    LIF::VarValues lifInit(
        -60.0,        // 0 - V
        0.0);       // 1 - RefracTime

    WeightUpdateModels::StaticPulse::VarValues dvsMacroPixelWeightUpdateInit(
        0.8);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues macroPixelOutputExcitatoryWeightUpdateInit(
        1.0);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues macroPixelOutputInhibitoryWeightUpdateInit(
        -0.5);     // 0 - Wij (nA)

    // Exponential current parameters
    ExpCurr::ParamValues macroPixelPostSynParams(
        5.0);         // 0 - TauSyn (ms)

    ExpCurr::ParamValues outputExcitatoryPostSynParams(
        25.0);         // 0 - TauSyn (ms)

    ExpCurr::ParamValues outputInhibitoryPostSynParams(
        50.0);         // 0 - TauSyn (ms)

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    // Create IF_curr neuron
    auto *dvs = model.addNeuronPopulation<NeuronModels::SpikeSource>("DVS", Parameters::inputSize * Parameters::inputSize,
                                                         {}, {});
    model.addNeuronPopulation<LIF>("MacroPixel", Parameters::macroPixelSize * Parameters::macroPixelSize,
                                   lifParams, lifInit);

    auto *output = model.addNeuronPopulation<LIF>("Output", Parameters::detectorSize * Parameters::detectorSize * Parameters::DetectorMax,
                                   lifParams, lifInit);

    //------------------------------------------------------------------------
    // Synapse populations
    //------------------------------------------------------------------------
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "DVS_MacroPixel", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "DVS", "MacroPixel",
        {}, dvsMacroPixelWeightUpdateInit,
        macroPixelPostSynParams, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "MacroPixel_Output_Excitatory", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "MacroPixel", "Output",
        {}, macroPixelOutputExcitatoryWeightUpdateInit,
        outputExcitatoryPostSynParams, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "MacroPixel_Output_Inhibitory", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "MacroPixel", "Output",
        {}, macroPixelOutputInhibitoryWeightUpdateInit,
        outputInhibitoryPostSynParams, {});

    // Use zero-copy for input and output spikes as we want to access them every timestep
    dvs->setSpikeZeroCopyEnabled(true);
    output->setSpikeZeroCopyEnabled(true);

    model.finalize();
}
