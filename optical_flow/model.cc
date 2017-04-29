#include <cmath>
#include <vector>

// GeNN includes
#include "modelSpec.h"

// Common example includes
#include "../common/exp_curr.h"
#include "../common/lif.h"

// LGMD includes
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
    LIF::ParamValues macroPixelParams(
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        1.0);    // 6 - TauRefrac

    // LIF initial conditions
    LIF::VarValues macroPixelInit(
        -60.0,        // 0 - V
        0.0);       // 1 - RefracTime

    WeightUpdateModels::StaticPulse::VarValues dvsMacroPixelWeightUpdateInit(
        0.8);     // 0 - Wij (nA)

    // Exponential current parameters
    ExpCurr::ParamValues dvsMacroPixelPostSynParams(
        5.0);         // 0 - TauSyn (ms)

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::SpikeSource>("DVS", Parameters::inputSize * Parameters::inputSize,
                                                         {}, {});
    model.addNeuronPopulation<LIF>("MacroPixel", Parameters::macroPixelSize * Parameters::macroPixelSize,
                                   macroPixelParams, macroPixelInit);

    //model.addNeuronPopulation<LIF>("LGMD", 1,
     //                              lgmd_lif_params, lif_init);

    //------------------------------------------------------------------------
    // Synapse populations
    //------------------------------------------------------------------------
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "DVS_MacroPixel", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "DVS", "MacroPixel",
        {}, dvsMacroPixelWeightUpdateInit,
        dvsMacroPixelPostSynParams, {});

    /*model.setSpanTypeToPre("EE");
    model.setSpanTypeToPre("EI");
    model.setSpanTypeToPre("II");
    model.setSpanTypeToPre("IE");*/

    // Use zero-copy for spikes and weights as we want to record them every timestep
    //e->setSpikeZeroCopyEnabled(true);
    //ie->setWUVarZeroCopyEnabled("g", true);

    model.finalize();
}