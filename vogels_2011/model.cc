#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "connectors.h"
#include "exp_curr.h"
#include "lif.h"
#include "../common/vogels_2011.h"


void modelDefinition(NNmodel &model)
{
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;

    initGeNN();
    model.setDT(1.0);
    model.setName("vogels_2011");
    model.setTiming(true);
    
    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitVarSnippet::Uniform::ParamValues vDist(
        -60.0,  // 0 - min
        -50.0); // 1 - max

    // LIF model parameters
    LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.2,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    LIF::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),    // 0 - V
        0.0);                                       // 1 - RefracTime

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues excitatoryStaticSynapseInit(
        0.03);     // 0 - Wij (nA)

    WeightUpdateModels::StaticPulse::VarValues inhibitoryStaticSynapseInit(
        -0.03);    // 0 - Wij (nA)

    // Additive STDP synapse parameters
    Vogels2011::ParamValues vogels2011AdditiveSTDPParams(
        20.0,   // 0 - Tau
        0.12,   // 1 - rho
        0.005,  // 2 - eta
        -1.0,    // 3 - Wmin
        0.0);    // 4 - Wmax

    Vogels2011::VarValues vogels2011AdditiveSTDPInit(
        0.0);  // 0 - g

    // Exponential current parameters
    ExpCurr::ParamValues excitatoryExpCurrParams(
        5.0);  // 0 - TauSyn (ms)

    ExpCurr::ParamValues inhibitoryExpCurrParams(
        10.0);  // 0 - TauSyn (ms)

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<LIF>("E", 2000, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<LIF>("I", 500, lifParams, lifInit);

    auto *ee = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "EE", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {});
    auto *ei = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "EI", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "E", "I",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {});
    auto *ii = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, ExpCurr>(
        "II", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "I", "I",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {});
    auto *ie = model.addSynapsePopulation<Vogels2011, ExpCurr>(
        "IE", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "I", "E",
        vogels2011AdditiveSTDPParams, vogels2011AdditiveSTDPInit,
        inhibitoryExpCurrParams, {});

    // Configure plastic weight variables they can be downloaded to host
    ie->setWUVarMode("g", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    i->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

    ee->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(2000, 2000, 0.02));
    ee->setMaxSourceConnections(calcFixedProbabilityConnectorMaxSourceConnections(2000, 2000, 0.02));
    
    ei->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(2000, 500, 0.02));
    ei->setMaxSourceConnections(calcFixedProbabilityConnectorMaxSourceConnections(2000, 500, 0.02));
    
    ii->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(500, 500, 0.02));
    ii->setMaxSourceConnections(calcFixedProbabilityConnectorMaxSourceConnections(500, 500, 0.02));
    
    ie->setMaxConnections(calcFixedProbabilityConnectorMaxConnections(500, 2000, 0.02));
    ie->setMaxSourceConnections(calcFixedProbabilityConnectorMaxSourceConnections(500, 2000, 0.02));

    model.finalize();
}