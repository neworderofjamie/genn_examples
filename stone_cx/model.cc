#include "modelSpec.h"

// Common includes
#include "../common/sigmoid.h"

// Stone CX includes
#include "parameters.h"

//---------------------------------------------------------------------------
// Continuous
//---------------------------------------------------------------------------
class Continuous : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Continuous, 0, 1);

    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE(
        "$(addtoinSyn) = $(g) * $(r_pre);\n"
        "$(updatelinsyn);\n");
};
IMPLEMENT_MODEL(Continuous);

//---------------------------------------------------------------------------
// TN2Linear
//---------------------------------------------------------------------------
class TN2Linear : public NeuronModels::Base
{
public:
    DECLARE_MODEL(TN2Linear,0, 2);

    // **NOTE** this comes from https://github.com/InsectRobotics/path-integration/blob/master/cx_rate.py#L170-L173 rather than the methods section
    SET_SIM_CODE(
        "const scalar iTN = (sin($(headingAngle) + $(preferredAngle)) * $(vX)) + \n"
        "   (cos($(headingAngle) + $(preferredAngle)) * $(vY));\n"
        "$(r) = min(1.0, max(iTN, 0.0));\n");

    SET_VARS({{"r", "scalar"},
              {"preferredAngle", "scalar"}});

    SET_EXTRA_GLOBAL_PARAMS({{"headingAngle", "scalar"},
                             {"vX", "scalar"}, {"vY", "scalar"}});
};
IMPLEMENT_MODEL(TN2Linear);

//---------------------------------------------------------------------------
// TLSigmoid
//---------------------------------------------------------------------------
class TLSigmoid : public NeuronModels::Base
{
public:
    DECLARE_MODEL(TLSigmoid, 2, 2);

    SET_SIM_CODE(
        "const scalar iTL = cos($(preferredAngle) - $(headingAngle));\n"
        "$(r) = 1.0 / (1.0 + exp(-(($(a) * iTL) - $(b))));\n"
    );

    SET_PARAM_NAMES({
        "a",        // Multiplicative scale
        "b"});      // Additive scale

    SET_VARS({{"r", "scalar"},
              {"preferredAngle", "scalar"}});

    SET_EXTRA_GLOBAL_PARAMS({{"headingAngle", "scalar"}});
};
IMPLEMENT_MODEL(TLSigmoid);

//----------------------------------------------------------------------------
// CPU4Sigmoid
//----------------------------------------------------------------------------
//! Non-spiking sigmoid unit
class CPU4Sigmoid : public NeuronModels::Base
{
public:
    DECLARE_MODEL(CPU4Sigmoid, 4, 2);

    SET_SIM_CODE(
        "$(i) += $(h) * min(1.0, max($(Isyn), 0.0));\n"
        "$(i) -= $(h) * $(k);\n"
        "$(i) = min(1.0, max($(i), 0.0));\n"
        "$(r) = 1.0 / (1.0 + exp(-(($(a) * $(i)) - $(b))));\n"
    );

    SET_PARAM_NAMES({
        "a",        // Multiplicative scale
        "b",        // Additive scale
        "h",        // Input scale
        "k"});      // Offset current

    SET_VARS({{"r", "scalar"},
              {"i", "scalar"}});
};
IMPLEMENT_MODEL(CPU4Sigmoid);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(1.0);
    model.setName("stone_cx");

    //---------------------------------------------------------------------------
    // Neuron parameters
    //---------------------------------------------------------------------------
    Sigmoid::VarValues sigmoidInit(0.0);

    // TN2
    TN2Linear::VarValues tn2Init(
        0.0,    // r
        0.0);   // Preference angle (radians)

    // TL
    TLSigmoid::ParamValues tlParams(
        6.8,    // Multiplicative scale
        3.0);   // Additive scale

    TLSigmoid::VarValues tlInit(
        0.0,    // r
        0.0);   // Preference angle (radians)

    // CL1
    Sigmoid::ParamValues cl1Params(
        3.0,     // Multiplicative scale
        -0.5);   // Additive scale

    // TB1
    Sigmoid::ParamValues tb1Params(
        5.0,    // Multiplicative scale
        0.0);   // Additive scale

    // CPU4
    CPU4Sigmoid::ParamValues cpu4Params(
        5.0,    // Multiplicative scale
        2.5,    // Additive scale
        0.0025, // Input scale
        0.125);   // Offset current **NOTE** this is the value from github

    CPU4Sigmoid::VarValues cpu4Init(
        0.0,    // r
        0.5);   // i

    // Pontine
    Sigmoid::ParamValues pontineParams(
        5.0,     // Multiplicative scale
        2.5);   // Additive scale

    // CPU1
    Sigmoid::ParamValues cpu1Params(
        6.0,     // Multiplicative scale
        2.0);   // Additive scale

    //---------------------------------------------------------------------------
    // Synapse parameters
    //---------------------------------------------------------------------------
    Continuous::VarValues continuousExcInit(1.0);

    Continuous::VarValues continuousInhInit(-1.0);

    Continuous::VarValues cl1TB1Init(1.0 - Parameters::c);
    Continuous::VarValues cpu4CPU1Init(0.5);
    Continuous::VarValues pontineCPU1Init(-0.5);

    //---------------------------------------------------------------------------
    // Neuron populations
    //---------------------------------------------------------------------------
    model.addNeuronPopulation<TN2Linear>("TN2", Parameters::numTN2, {}, tn2Init);
    model.addNeuronPopulation<TLSigmoid>("TL", Parameters::numTL, tlParams, tlInit);
    model.addNeuronPopulation<Sigmoid>("CL1", Parameters::numCL1, cl1Params, sigmoidInit);
    model.addNeuronPopulation<Sigmoid>("TB1", Parameters::numTB1, tb1Params, sigmoidInit);
    model.addNeuronPopulation<CPU4Sigmoid>("CPU4", Parameters::numCPU4, cpu4Params, cpu4Init);
    model.addNeuronPopulation<Sigmoid>("Pontine", Parameters::numPontine, pontineParams, sigmoidInit);
    model.addNeuronPopulation<Sigmoid>("CPU1", Parameters::numCPU1, cpu1Params, sigmoidInit);

    //---------------------------------------------------------------------------
    // Synapse populations
    //---------------------------------------------------------------------------
    auto *tlCL1 = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "TL_CL1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "TL", "CL1",
        {}, continuousInhInit,
        {}, {});

    auto *cl1TB1 = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "CL1_TB1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "CL1", "TB1",
        {}, cl1TB1Init,
        {}, {});

    auto *tb1TB1 = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "TB1_TB1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "TB1", "TB1",
        {}, continuousInhInit,
        {}, {});

    auto *cpu4Pontine = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "CPU4_Pontine", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "CPU4", "Pontine",
        {}, continuousExcInit,
        {}, {});

    auto *tb1CPU4 = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "TB1_CPU4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "TB1", "CPU4",
        {}, continuousInhInit,
        {}, {});

    auto *tb1CPU1 = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "TB1_CPU1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "TB1", "CPU1",
        {}, continuousInhInit,
        {}, {});

    auto *cpu4CPU1 = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "CPU4_CPU1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "CPU4", "CPU1",
        {}, cpu4CPU1Init,
        {}, {});

    auto *tn2CPU4 = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "TN2_CPU4", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "TN2", "CPU4",
        {}, continuousExcInit,
        {}, {});

    auto *pontineCPU1 = model.addSynapsePopulation<Continuous, PostsynapticModels::DeltaCurr>(
        "Pontine_CPU1", SynapseMatrixType::SPARSE_GLOBALG, NO_DELAY,
        "Pontine", "CPU1",
        {}, pontineCPU1Init,
        {}, {});

    // Finalize model
    model.finalize();
}