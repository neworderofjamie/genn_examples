#include <cmath>
#include <vector>

#include "binomial.h"
#include "modelSpec.h"

// GeNN robotics includes
#include "../common/vogels_2011.h"

void modelDefinition(NNmodel &model)
{
    model.setDT(1.0);
    model.setName("vogels_2011");
    model.setTiming(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    NeuronModels::LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.2,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        uninitialisedVar(), // 0 - V
        0.0);               // 1 - RefracTime

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
    PostsynapticModels::ExpCurr::ParamValues excitatoryExpCurrParams(
        5.0);  // 0 - TauSyn (ms)

    PostsynapticModels::ExpCurr::ParamValues inhibitoryExpCurrParams(
        10.0);  // 0 - TauSyn (ms)

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<NeuronModels::LIF>("E", 2000, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<NeuronModels::LIF>("I", 500, lifParams, lifInit);

    auto *ee = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "EE", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {});
    auto *ei = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "EI", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "E", "I",
        {}, excitatoryStaticSynapseInit,
        excitatoryExpCurrParams, {});
    auto *ii = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
        "II", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "I", "I",
        {}, inhibitoryStaticSynapseInit,
        inhibitoryExpCurrParams, {});
    auto *ie = model.addSynapsePopulation<Vogels2011, PostsynapticModels::ExpCurr>(
        "IE", SynapseMatrixType::SPARSE_INDIVIDUALG, NO_DELAY,
        "I", "E",
        vogels2011AdditiveSTDPParams, vogels2011AdditiveSTDPInit,
        inhibitoryExpCurrParams, {});

    ee->setMaxConnections(binomialInverseCDF(pow(0.9999, 1.0 / 2000.0), 2000, 0.02));
    ee->setMaxSourceConnections(binomialInverseCDF(pow(0.9999, 1.0 / 2000.0), 2000, 0.02));
    ei->setMaxConnections(binomialInverseCDF(pow(0.9999, 1.0 / 2000.0), 500, 0.02));
    ei->setMaxSourceConnections(binomialInverseCDF(pow(0.9999, 1.0 / 500.0), 2000, 0.02));
    ii->setMaxConnections(binomialInverseCDF(pow(0.9999, 1.0 / 500.0), 500, 0.02));
    ii->setMaxSourceConnections(binomialInverseCDF(pow(0.9999, 1.0 / 500.0), 500, 0.02));
    ie->setMaxConnections(binomialInverseCDF(pow(0.9999, 1.0 / 500.0), 2000, 0.02));
    ie->setMaxSourceConnections(binomialInverseCDF(pow(0.9999, 1.0 / 2000.0), 500, 0.02));
    
    // Configure plastic weight variables they can be downloaded to host
    ie->setWUVarLocation("g", VarLocation::HOST_DEVICE);
    ie->setSparseConnectivityLocation(VarLocation::HOST_DEVICE);

    // Configure spike variables so that they can be downloaded to host
    e->setSpikeLocation(VarLocation::HOST_DEVICE);
    e->setSpikeTimeLocation(VarLocation::HOST_DEVICE);
    i->setSpikeLocation(VarLocation::HOST_DEVICE);
}
