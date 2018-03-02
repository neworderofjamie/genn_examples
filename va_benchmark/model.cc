#include <cmath>
#include <vector>

#include "modelSpec.h"

// GeNN robotics includes
#include "connectors.h"
#include "exp_curr.h"
#include "lif.h"

#include "parameters.h"


void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(1.0);
    model.setName("va_benchmark");

    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitVarSnippet::Uniform::ParamValues vDist(
        Parameters::resetVoltage,       // 0 - min
        Parameters::thresholdVoltage);  // 1 - max

    // LIF model parameters
    LIF::ParamValues lifParams(
        1.0,    // 0 - C
        20.0,   // 1 - TauM
        -49.0,  // 2 - Vrest
        Parameters::resetVoltage,  // 3 - Vreset
        Parameters::thresholdVoltage,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    LIF::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),     // 0 - V
        0.0);   // 1 - RefracTime

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues excitatoryStaticSynapseInit(
        Parameters::excitatoryWeight);    // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues inhibitoryStaticSynapseInit(
        Parameters::inhibitoryWeight);    // 0 - Wij (nA)

    // Exponential current parameters
    ExpCurr::ParamValues excitatoryExpCurrParams(
        5.0);  // 0 - TauSyn (ms)

    ExpCurr::ParamValues inhibitoryExpCurrParams(
        10.0);  // 0 - TauSyn (ms)

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<LIF>("E", Parameters::numExcitatory, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<LIF>("I", Parameters::numInhibitory, lifParams, lifInit);

    auto *ee = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "EE", SynapseMatrixType::BITMASK_GLOBALG, NO_DELAY,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {});
    auto *ei = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "EI", SynapseMatrixType::BITMASK_GLOBALG, NO_DELAY,
        "E", "I",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {});
    auto *ii = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "II", SynapseMatrixType::BITMASK_GLOBALG, NO_DELAY,
        "I", "I",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {});
    auto *ie = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "IE", SynapseMatrixType::BITMASK_GLOBALG, NO_DELAY,
        "I", "E",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {});

    /*ee->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(Parameters::numExcitatory, Parameters::numExcitatory,
                                                                      Parameters::probabilityConnection));
    ei->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(Parameters::numExcitatory, Parameters::numInhibitory,
                                                                      Parameters::probabilityConnection));
    ii->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(Parameters::numInhibitory, Parameters::numInhibitory,
                                                                      Parameters::probabilityConnection));
    ie->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(Parameters::numInhibitory, Parameters::numExcitatory,
                                                                      Parameters::probabilityConnection));*/

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    i->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    model.finalize();
}