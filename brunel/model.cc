#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "parameters.h"

class EulerLIF : public NeuronModels::Base
{
public:
    DECLARE_MODEL(EulerLIF, 6, 2);

    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0) {\n"
        "  $(V) += (DT / $(TauM))*(($(Vrest) - $(V)) + $(Ioffset)) + $(Isyn);\n"
        "}\n"
        "else {\n"
        "  $(RefracTime) -= DT;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "Ioffset",    // Offset current
        "TauRefrac"});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}});
};
IMPLEMENT_MODEL(EulerLIF);

class STDPExponential : public WeightUpdateModels::Base
{
public:
    DECLARE_WEIGHT_UPDATE_MODEL(STDPExponential, 5, 1, 1, 1);
    SET_PARAM_NAMES({"tauSTDP", "alpha", "lambda",
                     "Wmin", "Wmax"});
    SET_DERIVED_PARAMS({
        {"tauSTDPDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});
    SET_VARS({{"g", "scalar"}});
    SET_PRE_VARS({{"preTrace", "scalar"}});
    SET_POST_VARS({{"postTrace", "scalar"}});

    SET_SIM_CODE(
        "$(addToInSyn, $(g));\n"
        "const scalar newWeight = $(g) - ($(alpha) * $(lambda) * $(g) * exp(-$(postTrace) / DT));\n"
        "$(g) = fmax($(Wmin), newWeight);\n");
    SET_LEARN_POST_CODE(
        "const scalar newWeight = $(g) + ($(lambda) * (1.0 - $(g)) * exp(-$(preTrace) / DT));\n"
        "$(g) = fmin($(Wmax), newWeight);\n");
    SET_PRE_SPIKE_CODE("$(preTrace) += 1.0;\n");
    SET_POST_SPIKE_CODE("$(postTrace) += 1.0;\n");
    SET_PRE_DYNAMICS_CODE("$(preTrace) *= $(tauSTDPDecay);\n");
    SET_POST_DYNAMICS_CODE("$(postTrace) *= $(tauSTDPDecay);\n");
};
IMPLEMENT_MODEL(STDPExponential);

void modelDefinition(NNmodel &model)
{
    // **NOTE** in the absence of a better system, manually "caching" these in code after running genn-buildmodel once on a particular GPU speeds up build time hugely
    /*GENN_PREFERENCES.deviceSelectMethod = DeviceSelect::MOST_MEMORY;
    GENN_PREFERENCES.blockSizeSelectMethod = BlockSizeSelect::MANUAL;
    GENN_PREFERENCES.manualBlockSizes[CodeGenerator::KernelNeuronUpdate] = 64;
    GENN_PREFERENCES.manualBlockSizes[CodeGenerator::KernelPresynapticUpdate] = 32;
    GENN_PREFERENCES.manualBlockSizes[CodeGenerator::KernelPostsynapticUpdate] = 32;
    GENN_PREFERENCES.manualBlockSizes[CodeGenerator::KernelInitialize] = 32;
    GENN_PREFERENCES.manualBlockSizes[CodeGenerator::KernelInitializeSparse] = 64;
    GENN_PREFERENCES.manualBlockSizes[CodeGenerator::KernelNeuronSpikeQueueUpdate] = 32;*/

    // Use approximate exponentials etc to speed up plasticity
    GENN_PREFERENCES.optimizeCode = true;

    model.setDT(Parameters::timestep);
    model.setName("brunel");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTiming(true);
    model.setMergePostsynapticModels(true);
    model.setSeed(1234);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitVarSnippet::Uniform::ParamValues vDist(
        Parameters::resetVoltage,       // 0 - min
        Parameters::thresholdVoltage);  // 1 - max

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(
        Parameters::probabilityConnection); // 0 - prob

    // LIF model parameters
    EulerLIF::ParamValues lifParams(
        20.0,                           // 0 - TauM
        Parameters::resetVoltage,       // 1 - Vrest
        Parameters::resetVoltage,       // 2 - Vreset
        Parameters::thresholdVoltage,   // 3 - Vthresh
        0.0,                            // 4 - Ioffset
        2.0);                           // 5 - TauRefrac

    // LIF initial conditions
    EulerLIF::VarValues lifInit(
        0.0,    // 0 - V
        0.0);   // 1 - RefracTime

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues excitatoryStaticSynapseInit(
        Parameters::excitatoryWeight);    // 0 - Wij (mV)

    WeightUpdateModels::StaticPulse::VarValues inhibitoryStaticSynapseInit(
        Parameters::inhibitoryWeight);    // 0 - Wij (mV)

    // STDP parameters
    STDPExponential::ParamValues stdpParams(
        20.0,   // tauSTDP (ms)
        2.02,   // alpha
        0.01,   // lambda
        0.0,    // Wmin (mV)
        0.3);   // Wmax (mV)
    STDPExponential::VarValues stdpInit(
        Parameters::excitatoryWeight);  // 0 - Wij (mV)

    NeuronModels::PoissonNew::ParamValues poissonParams(Parameters::inputRate); // 0 - rate (Hz)
    NeuronModels::PoissonNew::VarValues poissonInit(0.0);                       // 0 - timeStepToSpike

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<EulerLIF>("E", Parameters::numExcitatory, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<EulerLIF>("I", Parameters::numInhibitory, lifParams, lifInit);

    auto *poisson = model.addNeuronPopulation<NeuronModels::PoissonNew>("Poisson", Parameters::numNeurons,
                                                                        poissonParams, poissonInit);

    // Enable spike recording
    e->setSpikeRecordingEnabled(true);
    i->setSpikeRecordingEnabled(true);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "PoissonE", SynapseMatrixType::SPARSE_GLOBALG, Parameters::delayTimesteps,
        "Poisson", "E",
        {}, excitatoryStaticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "PoissonI", SynapseMatrixType::SPARSE_GLOBALG, Parameters::delayTimesteps,
        "Poisson", "I",
        {}, excitatoryStaticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

#ifdef STDP
    model.addSynapsePopulation<STDPExponential, PostsynapticModels::DeltaCurr>(
        "EE", SynapseMatrixType::SPARSE_INDIVIDUALG, Parameters::delayTimesteps,
        "E", "E",
        stdpParams, stdpInit, {0.0}, {0.0},
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
#else
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "EE", SynapseMatrixType::SPARSE_GLOBALG, Parameters::delayTimesteps,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
#endif

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "EI", SynapseMatrixType::SPARSE_GLOBALG, Parameters::delayTimesteps,
        "E", "I",
        {}, excitatoryStaticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "II", SynapseMatrixType::SPARSE_GLOBALG, Parameters::delayTimesteps,
        "I", "I",
        {}, inhibitoryStaticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "IE", SynapseMatrixType::SPARSE_GLOBALG, Parameters::delayTimesteps,
        "I", "E",
        {}, inhibitoryStaticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));
}
