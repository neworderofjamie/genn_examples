// Model definintion file tenHHRing.cc
#include "modelSpec.h"

class Ring : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(Ring);
    SET_ROW_BUILD_CODE(
        "addSynapse((id_pre + 1) % num_post);\n");
    SET_CALC_MAX_ROW_LENGTH_FUNC([](unsigned int, unsigned int, const std::unordered_map<std::string,double> &){ return 1;});
};
IMPLEMENT_SNIPPET(Ring);

void modelDefinition(ModelSpec &model)
{
    // definition of tenHHRing
    model.setDT(0.1);
    model.setName("tenHHRing");

    ParamValues p{
        {"gNa", 7.15},      // 0 - gNa: Na conductance in muS
        {"ENa", 50.0},      // 1 - ENa: Na equi potential in mV
        {"gK", 1.43},       // 2 - gK: K conductance in muS
        {"EK", -95.0},      // 3 - EK: K equi potential in mV
        {"gl", 0.02672},    // 4 - gl: leak conductance in muS
        {"El", -63.563},    // 5 - El: leak equi potential in mV
        {"C", 0.143}};      // 6 - C: membr. capacity density in nF

    VarValues ini{
        {"V", -60.0},         // 0 - membrane potential V
        {"m", 0.0529324},     // 1 - prob. for Na channel activation m
        {"h", 0.3176767},     // 2 - prob. for not Na channel blocking h
        {"n", 0.5961207}};    // 3 - prob. for K channel activation n

    VarValues stim_ini{
        {"startSpike", uninitialisedVar()},     // 0 - startSpike indices
        {"endSpike", uninitialisedVar()}};    // 1 - endSpike indices

    auto *pop1 = model.addNeuronPopulation<NeuronModels::TraubMiles>("Pop1", 10, p, ini);
    model.addNeuronPopulation<NeuronModels::SpikeSourceArray>("Stim", 1, {}, stim_ini);

    ParamValues s_ini{{"g",  -0.2}};

    ParamValues ps_p{{"tau", 1.0}, {"E", -80.0}};

    model.addSynapsePopulation(
        "Pop1self", SynapseMatrixType::SPARSE, 100,
        "Pop1", "Pop1",
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(s_ini),
        initPostsynaptic<PostsynapticModels::ExpCond>(ps_p, {}, {{"V", createVarRef(pop1, "V")}}),
        initConnectivity<Ring>());

    model.addSynapsePopulation(
        "StimPop1", SynapseMatrixType::SPARSE, NO_DELAY,
        "Stim", "Pop1",
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(s_ini),
        initPostsynaptic<PostsynapticModels::ExpCond>(ps_p, {}, {{"V", createVarRef(pop1, "V")}}),
        initConnectivity<InitSparseConnectivitySnippet::OneToOne>());
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
    runtime.allocate();
    runtime.initialize();
       
    auto *stim = model.findNeuronGroup("Stim");
    runtime.getArray(*stim, "startSpike")->getHostPointer<unsigned int>()[0] = 0;
    runtime.getArray(*stim, "endSpike")->getHostPointer<unsigned int>()[0] = 1;
    runtime.initializeSparse();
    
    runtime.allocateArray(*stim, "spikeTimes", 1);
    auto *spikeTimes = runtime.getArray(*stim, "spikeTimes");
    spikeTimes->getHostPointer<float>()[0] = 0.0f;
    spikeTimes->pushToDevice();

    auto *pop = model.findNeuronGroup("Pop1");
    auto *v = runtime.getArray(*pop, "V");
    std::ofstream os("tenHHRing_output.V.dat");
    while(runtime.getTime() < 200.0f) {
        runtime.stepTime();
        v->pullFromDevice();

        os << runtime.getTime() << " ";
        for (int j= 0; j < 10; j++) {
            os << v->getHostPointer<float>()[j] << " ";
        }
        os << std::endl;
    }
    os.close();
}
