#include <cmath>
#include <vector>

#include "modelSpec.h"

#include "parameters.h"

class PoissonDelta : public CurrentSourceModels::Base
{
    DECLARE_MODEL(PoissonDelta, 2, 0);

    SET_INJECTION_CODE(
        "scalar p = 1.0f;\n"
        "unsigned int numSpikes = 0;\n"
        "do\n"
        "{\n"
        "    numSpikes++;\n"
        "    p *= $(gennrand_uniform);\n"
        "} while (p > $(ExpMinusLambda));\n"
        "$(injectCurrent, $(weight) * (scalar)(numSpikes - 1));\n");

    SET_PARAM_NAMES({"weight", "rate"});
    SET_DERIVED_PARAMS({
        {"ExpMinusLambda", [](const std::vector<double> &pars, double dt){ return std::exp(-(pars[1] / 1000.0) * dt); }}});
};
IMPLEMENT_MODEL(PoissonDelta);

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
        "const scalar dt = $(t) - $(sT_post); \n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) - ($(alpha) * $(lambda) * $(g) * exp(-$(postTrace)));\n"
        "    $(g) = fmax($(Wmin), newWeight);\n"
        "}\n");
    SET_LEARN_POST_CODE(
        "const scalar dt = $(t) - $(sT_pre);\n"
        "if (dt > 0) {\n"
        "    const scalar newWeight = $(g) + ($(lambda) * (1.0 - $(g)) * exp(-$(preTrace)));\n"
        "    $(g) = fmin($(Wmax), newWeight);\n"
        "}\n");
    SET_PRE_SPIKE_CODE("$(preTrace) += DT;\n");
    SET_POST_SPIKE_CODE("$(postTrace) += DT;\n");
    SET_PRE_DYNAMICS_CODE("$(preTrace) *= $(tauSTDPDecay);\n");
    SET_POST_DYNAMICS_CODE("$(postTrace) *= $(tauSTDPDecay);\n");
    
    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_POST_SPIKE_TIME(true);
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
    NeuronModels::LIF::ParamValues lifParams(
        1.0E-3,                         // 0 - C
        20.0,                           // 1 - TauM
        Parameters::resetVoltage,       // 2 - Vrest
        Parameters::resetVoltage,       // 3 - Vreset
        Parameters::thresholdVoltage,   // 4 - Vthresh
        0.0,                            // 5 - Ioffset
        2.0);                           // 6 - TauRefrac

    // LIF initial conditions
    // **NOTE** not 100% sure how voltages should be initialised with this model
    NeuronModels::LIF::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),     // 0 - V
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
        
    // Poisson input parameters
    PoissonDelta::ParamValues poissonParams(
        Parameters::excitatoryWeight,   // 0 - Weight (mV)
        20.0);                          // 1 - rate (Hz)

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<NeuronModels::LIF>("E", Parameters::numExcitatory, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<NeuronModels::LIF>("I", Parameters::numInhibitory, lifParams, lifInit);
    
    // Add Poisson current injection
    model.addCurrentSource<PoissonDelta>("EExt", "E", poissonParams, {});
    model.addCurrentSource<PoissonDelta>("IExt", "I", poissonParams, {});
    
    // Enable spike recording
    e->setSpikeRecordingEnabled(true);
    i->setSpikeRecordingEnabled(true);

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
