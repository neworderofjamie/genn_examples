// Model definintion file tenHHModel.cc
#include "modelSpec.h"

void modelDefinition(NNmodel &model)
{
    // definition of tenHHModel
    initGeNN();
    model.setDT(0.1);
    model.setName("tenHHModel");

    NeuronModels::TraubMiles::ParamValues p(
        7.15,          // 0 - gNa: Na conductance in muS
        50.0,          // 1 - ENa: Na equi potential in mV
        1.43,          // 2 - gK: K conductance in muS
        -95.0,          // 3 - EK: K equi potential in mV
        0.02672,       // 4 - gl: leak conductance in muS
        -63.563,       // 5 - El: leak equi potential in mV
        0.143);       // 6 - Cmem: membr. capacity density in nF

    NeuronModels::TraubMiles::VarValues ini(
        -60.0,         // 0 - membrane potential V
        0.0529324,     // 1 - prob. for Na channel activation m
        0.3176767,     // 2 - prob. for not Na channel blocking h
        0.5961207);    // 3 - prob. for K channel activation n

    model.addNeuronPopulation<NeuronModels::TraubMiles>("Pop1", 10, p, ini);
    model.finalize();
}