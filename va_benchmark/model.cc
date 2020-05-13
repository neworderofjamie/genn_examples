#include <cmath>
#include <vector>

#include "binomial.h"
#include "modelSpec.h"

#include "parameters.h"

void modelDefinition(NNmodel &model)
{
    model.setDT(1.0);
    model.setName("va_benchmark");
    model.setTiming(false);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    NeuronModels::LIF::ParamValues lifParams(
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        -49.0,  // 2 - Vrest
        Parameters::resetVoltage,  // 3 - Vreset
        Parameters::thresholdVoltage,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        uninitialisedVar(),     // 0 - V
        0.0);   // 1 - RefracTime

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
    auto *e = model.addNeuronPopulation<NeuronModels::LIF>("E", Parameters::numExcitatory, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<NeuronModels::LIF>("I", Parameters::numInhibitory, lifParams, lifInit);

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeLocation(VarLocation::HOST_DEVICE);
    i->setSpikeLocation(VarLocation::HOST_DEVICE);

    // Determine matrix type
    const SynapseMatrixType matrixType = Parameters::bitmaskConnectivity ? SynapseMatrixType::BITMASK_GLOBALG : SynapseMatrixType::SPARSE_GLOBALG;

    auto *ee = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "EE", matrixType, NO_DELAY,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {});
    auto *ei = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "EI", matrixType, NO_DELAY,
        "E", "I",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {});
    auto *ii = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "II", matrixType, NO_DELAY,
        "I", "I",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {});
    auto *ie = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "IE", matrixType, NO_DELAY,
        "I", "E",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {});

    ee->setMaxConnections(binomialInverseCDF(pow(0.9999, 1.0 / (double)Parameters::numExcitatory), Parameters::numExcitatory, Parameters::probabilityConnection));
    ei->setMaxConnections(binomialInverseCDF(pow(0.9999, 1.0 / (double)Parameters::numExcitatory), Parameters::numInhibitory, Parameters::probabilityConnection));
    ii->setMaxConnections(binomialInverseCDF(pow(0.9999, 1.0 / (double)Parameters::numInhibitory), Parameters::numInhibitory, Parameters::probabilityConnection));
    ie->setMaxConnections(binomialInverseCDF(pow(0.9999, 1.0 / (double)Parameters::numInhibitory), Parameters::numExcitatory, Parameters::probabilityConnection));
            
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
