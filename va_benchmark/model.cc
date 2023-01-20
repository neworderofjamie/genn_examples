#include <cmath>
#include <vector>

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
    VarValues excitatoryStaticSynapseInit{{"g", Parameters::excitatoryWeight}};
    VarValues inhibitoryStaticSynapseInit{{"g", Parameters::inhibitoryWeight}};

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
        ? SynapseMatrixType::PROCEDURAL_GLOBALG
        : (Parameters::bitmaskConnectivity ? SynapseMatrixType::BITMASK_GLOBALG : SynapseMatrixType::SPARSE_GLOBALG);

    auto *ee = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "EE", matrixType, NO_DELAY,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    auto *ei = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "EI", matrixType, NO_DELAY,
        "E", "I",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    auto *ii = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "II", matrixType, NO_DELAY,
        "I", "I",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    auto *ie = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "IE", matrixType, NO_DELAY,
        "I", "E",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {},
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
