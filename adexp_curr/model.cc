#include "modelSpec.h"

// GeNN robotics includes
#include "../common/adexp.h"

void modelDefinition(NNmodel &model)
{
    model.setDT(0.1);
    model.setName("adexp_curr");

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // AdExp model parameters
    AdExp::ParamValues adExpParamVals(
        281.0,    // Membrane capacitance [pF]
        30.0,     // Leak conductance [nS]
        -70.6,    // Leak reversal potential [mV]
        2.0,      // Slope factor [mV]
        -50.4,    // Threshold voltage [mV]
        10.0,     // Artificial spike height [mV]
        -70.6,    // Reset voltage [mV]
        144.0,    // Adaption time constant
        4.0,      // Subthreshold adaption [nS]
        0.0805,   // Spike-triggered adaptation [nA]
        700.0);   // Offset current
    
    AdExp::VarValues adExpInitVals(
        -70.6,       // 0 - V
        0.0);        // 1 - W

    // Create IF_curr neuron
    model.addNeuronPopulation<AdExp>("Neurons", 1, adExpParamVals, adExpInitVals);
}
