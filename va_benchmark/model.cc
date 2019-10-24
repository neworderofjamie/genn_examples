#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "parameters.h"

void modelDefinition(NNmodel &model)
{
    model.setDT(1.0);
    model.setName("va_benchmark");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTiming(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitVarSnippet::Uniform::ParamValues vDist(
        Parameters::resetVoltage,       // 0 - min
        Parameters::thresholdVoltage);  // 1 - max

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(
        Parameters::probabilityConnection); // 0 - prob

    // LIF model parameters
    NeuronModels::LIFAuto::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),     // 0 - V
        0.0,   // 1 - RefracTime
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        -49.0,  // 2 - Vrest
        Parameters::resetVoltage,  // 3 - Vreset
        Parameters::thresholdVoltage,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac


    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues excitatoryStaticSynapseInit(
        Parameters::excitatoryWeight);    // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues inhibitoryStaticSynapseInit(
        Parameters::inhibitoryWeight);    // 0 - Wij (nA)

    // Exponential current parameters
    PostsynapticModels::ExpCurr::ParamValues excitatoryExpCurrParams(
        5.0);  // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurr::ParamValues inhibitoryExpCurrParams(
        10.0);  // 0 - TauSyn (ms)

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<NeuronModels::LIFAuto>("E", Parameters::numExcitatory, lifInit);
    auto *i = model.addNeuronPopulation<NeuronModels::LIFAuto>("I", Parameters::numInhibitory, lifInit);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "EE", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "EI", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "E", "I",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "II", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "I", "I",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "IE", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "I", "E",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeLocation(VarLocation::HOST_DEVICE);
    i->setSpikeLocation(VarLocation::HOST_DEVICE);
}
