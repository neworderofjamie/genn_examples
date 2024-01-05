#include <fstream>

#include "modelSpec.h"

#include "parameters.h"

void modelDefinition(ModelSpec &model)
{
    model.setDT(1.0);
    model.setName("va_benchmark");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTiming(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    ParamValues vDist{
        {"min", Parameters::resetVoltage},
        {"max", Parameters::thresholdVoltage}};

    ParamValues fixedProb{{"prob", Parameters::probabilityConnection}};

    // LIF model parameters
    ParamValues lifParams{
        {"C", 1.0},
        {"TauM", 20.0},
        {"Vrest", -49.0},
        {"Vreset", Parameters::resetVoltage},
        {"Vthresh", Parameters::thresholdVoltage},
        {"Ioffset", 0.0},
        {"TauRefrac", 5.0}};

    // LIF initial conditions
    VarValues lifInit{
        {"V", initVar<InitVarSnippet::Uniform>(vDist)},
        {"RefracTime", 0.0}};
    
    // Static synapse parameters
    ParamValues excitatoryStaticSynapseInit{{"g", Parameters::excitatoryWeight}};
    ParamValues inhibitoryStaticSynapseInit{{"g", Parameters::inhibitoryWeight}};

    // Exponential current parameters
    ParamValues excitatoryExpCurrParams{{"tau", 5.0}};
    ParamValues inhibitoryExpCurrParams{{"tau", 10.0}};

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<NeuronModels::LIF>("E", Parameters::numExcitatory, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<NeuronModels::LIF>("I", Parameters::numInhibitory, lifParams, lifInit);

    // Enable spike recording
    e->setSpikeRecordingEnabled(true);
    i->setSpikeRecordingEnabled(true);

    // Determine matrix type
    const SynapseMatrixType matrixType = Parameters::proceduralConnectivity
        ? SynapseMatrixType::PROCEDURAL
        : (Parameters::bitmaskConnectivity ? SynapseMatrixType::BITMASK : SynapseMatrixType::SPARSE);

    auto *ee = model.addSynapsePopulation(
        "EE", matrixType, e, e,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(excitatoryStaticSynapseInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(excitatoryExpCurrParams),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    auto *ei = model.addSynapsePopulation(
        "EI", matrixType, e, i,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(excitatoryStaticSynapseInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(excitatoryExpCurrParams),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    auto *ii = model.addSynapsePopulation(
        "II", matrixType, i, i,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(inhibitoryStaticSynapseInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(inhibitoryExpCurrParams),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    auto *ie = model.addSynapsePopulation(
        "IE", matrixType, i, e,
        initWeightUpdate<WeightUpdateModels::StaticPulseConstantWeight>(inhibitoryStaticSynapseInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(inhibitoryExpCurrParams),
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    if(Parameters::presynapticParallelism) {
        // Set span type
        ee->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);
        ei->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);
        ii->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);
        ie->setSpanType(SynapseGroup::SpanType::PRESYNAPTIC);

        // Set threads per spike
        ee->setNumThreadsPerSpike(Parameters::numThreadsPerSpike);
        ei->setNumThreadsPerSpike(Parameters::numThreadsPerSpike);
        ii->setNumThreadsPerSpike(Parameters::numThreadsPerSpike);
        ie->setNumThreadsPerSpike(Parameters::numThreadsPerSpike);
    }
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
    runtime.allocate(Parameters::numTimesteps);
    runtime.initialize();
    runtime.initializeSparse();
    
    const auto startTime = std::chrono::high_resolution_clock::now();
    while(runtime.getTimestep() < Parameters::numTimesteps) {
        runtime.stepTime();
    }
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - startTime;
    std::cout << "Init time:" << runtime.getInitTime() << std::endl;
    std::cout << "Total simulation time:" << duration.count() << " seconds" << std::endl;
    std::cout << "\tNeuron update time:" << runtime.getNeuronUpdateTime() << std::endl;
    std::cout << "\tPresynaptic update time:" << runtime.getPresynapticUpdateTime() << std::endl;
    
    runtime.pullRecordingBuffersFromDevice();
    
    const auto *e = model.findNeuronGroup("E");
    const auto *i = model.findNeuronGroup("I");
    runtime.writeRecordedSpikes(*e, "spikes_e.csv");
    runtime.writeRecordedSpikes(*i, "spikes_i.csv");

}

