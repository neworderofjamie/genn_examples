#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "parameters.h"

class StaticPulseHalf : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(StaticPulseHalf, 0, 1, 0, 0);

    SET_VARS({{"g", "half", VarAccess::READ_ONLY}});

    SET_SIM_CODE("$(addToInSynVec, $(g.x), $(g.y));\n");
};
IMPLEMENT_MODEL(StaticPulseHalf);

//typedef StaticPulseHalf WUM;
typedef WeightUpdateModels::StaticPulse WUM;

void modelDefinition(NNmodel &model)
{
    model.setDT(1.0);
    model.setName("va_benchmark");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTiming(true);
    //model.setDefaultNarrowSparseIndEnabled(true);

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
    WUM::VarValues excitatoryStaticSynapseInit(
        Parameters::excitatoryWeight);    // 0 - Wij (nA)

    WUM::VarValues inhibitoryStaticSynapseInit(
        Parameters::inhibitoryWeight);    // 0 - Wij (nA)

    // Exponential current parameters
    PostsynapticModels::ExpCurrAuto::VarValues excitatoryExpCurrInit(
        5.0);  // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurrAuto::VarValues inhibitoryExpCurrInit(
        10.0);  // 0 - TauSyn (ms)

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<NeuronModels::LIFAuto>("E", Parameters::numExcitatory, lifInit);
    auto *i = model.addNeuronPopulation<NeuronModels::LIFAuto>("I", Parameters::numInhibitory, lifInit);

    model.addSynapsePopulation<WUM, PostsynapticModels::ExpCurrAuto>(
        "EE", SynapseMatrixConnectivity::SPARSE, NO_DELAY,
        "E", "E",
        excitatoryStaticSynapseInit, excitatoryExpCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    model.addSynapsePopulation<WUM, PostsynapticModels::ExpCurrAuto>(
        "EI", SynapseMatrixConnectivity::SPARSE, NO_DELAY,
        "E", "I",
        excitatoryStaticSynapseInit, excitatoryExpCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WUM, PostsynapticModels::ExpCurrAuto>(
        "II", SynapseMatrixConnectivity::SPARSE, NO_DELAY,
        "I", "I",
        inhibitoryStaticSynapseInit, inhibitoryExpCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    model.addSynapsePopulation<WUM, PostsynapticModels::ExpCurrAuto>(
        "IE", SynapseMatrixConnectivity::SPARSE, NO_DELAY,
        "I", "E",
        inhibitoryStaticSynapseInit, inhibitoryExpCurrInit,
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeLocation(VarLocation::HOST_DEVICE);
    i->setSpikeLocation(VarLocation::HOST_DEVICE);
}
