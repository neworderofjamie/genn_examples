#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "../common/exp_curr.h"
#include "../common/lif.h"

#include "parameters.h"

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(1.0);
    model.setName("benchmark");


    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    // **TODO** uniform random
    LIF::VarValues lifInit(
        -55.0,  // 0 - V
        0.0);    // 1 - RefracTime

    NeuronModels::Poisson::ParamValues poissonParams(
        10.0,        // 0 - firing rate
        2.5,        // 1 - refratory period
        20.0,       // 2 - Vspike
        -60.0);       // 3 - Vrest

    NeuronModels::Poisson::VarValues poissonInit(
        -60.0,        // 0 - V
        0,           // 1 - seed
        -10.0);     // 2 - SpikeTime

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        0.0);    // 0 - Wij (nA)

    // Exponential current parameters
    ExpCurr::ParamValues expCurrParams(
        5.0);  // 0 - TauSyn (ms)

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::Poisson>("Stim", Parameters::numPre,
                                poissonParams, poissonInit);
    model.addNeuronPopulation<LIF>("Neurons", Parameters::numPost,
                                   lifParams, lifInit);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>("Syn", Parameters::synapseMatrixType, NO_DELAY,
                             "Stim", "Neurons",
                             {}, staticSynapseInit,
                             expCurrParams, {});

    model.finalize();
}