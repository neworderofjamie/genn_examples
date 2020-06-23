// GeNN includes
#include "modelSpec.h"

void modelDefinition(NNmodel &model)
{
    model.setDT(1.0);
    model.setName("spike_source_array");
    
    NeuronModels::SpikeSourceArray::VarValues ssaInit(
        uninitialisedVar(),     // 0 - startSpike
        uninitialisedVar());    // 1 - endSpike
    
    model.addNeuronPopulation<NeuronModels::SpikeSourceArray>(
        "SSA", 100, {}, ssaInit);
}