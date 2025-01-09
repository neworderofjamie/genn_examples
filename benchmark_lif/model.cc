#include <fstream>

#include "modelSpec.h"

class LIFHalf : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(LIFHalf);

    SET_SIM_CODE(
        "if (RefracTime <= 0.0) {\n"
        "  scalar alpha = ((Isyn + Ioffset) * Rmembrane) + Vrest;\n"
        "  V = alpha - (ExpTC * (alpha - V));\n"
        "}\n"
        "else {\n"
        "  RefracTime -= dt;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("RefracTime <= 0.0 && V >= Vthresh");

    SET_RESET_CODE(
        "V = Vreset;\n"
        "RefracTime = TauRefrac;\n");

    SET_PARAMS({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "Ioffset",    // Offset current
        "TauRefrac"});

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauM").cast<double>()); }},
        {"Rmembrane", [](const ParamValues &pars, double){ return  pars.at("TauM").cast<double>() / pars.at("C").cast<double>(); }}});

    SET_VARS({{"V", "scalar", "half"}, {"RefracTime", "scalar", "half"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_SNIPPET(LIFHalf);

void modelDefinition(ModelSpec &model)
{
    model.setDT(1.0);
    model.setName("benchmark_lif");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTimingEnabled(true);
    
    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    ParamValues lifParams{
        {"C", 1.0},
        {"TauM", 20.0},
        {"Vrest", -49.0},
        {"Vreset", -60.0},
        {"Vthresh", -50.0},
        {"Ioffset", 0.0},
        {"TauRefrac", 5.0}};

    // LIF initial conditions
    VarValues lifInit{
        {"V", -60.0},
        {"RefracTime", 0.0}};

    // Create IF_curr neuron
    auto *pop = model.addNeuronPopulation<LIFHalf>("Pop", 1000, lifParams, lifInit);

    // Enable spike recording
    pop->setSpikeRecordingEnabled(true);
    
    // Add current source
    model.addCurrentSource<CurrentSourceModels::GaussianNoise>("Noise", pop, {{"mean", 0.0}, {"sd", 1.0}});
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
    runtime.allocate(1000);
    runtime.initialize();
    runtime.initializeSparse();
    
    const auto startTime = std::chrono::high_resolution_clock::now();
    while(runtime.getTimestep() < 1000) {
        runtime.stepTime();
    }
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - startTime;
    std::cout << "Init time:" << runtime.getInitTime() << std::endl;
    std::cout << "Total simulation time:" << duration.count() << " seconds" << std::endl;
    std::cout << "\tNeuron update time:" << runtime.getNeuronUpdateTime() << std::endl;

    runtime.pullRecordingBuffersFromDevice();
    
    const auto *pop = model.findNeuronGroup("Pop");
    runtime.writeRecordedSpikes(*pop, "spikes_pop.csv");
}

