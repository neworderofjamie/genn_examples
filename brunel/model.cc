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

void modelDefinition(NNmodel &model)
{
    model.setDT(Parameters::timestep);
    model.setName("brunel");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTiming(true);
    model.setMergePostsynapticModels(true);
    
    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitVarSnippet::Uniform::ParamValues vDist(
        Parameters::resetVoltage,       // 0 - min
        Parameters::thresholdVoltage);  // 1 - max

    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(
        Parameters::probabilityConnection); // 0 - prob

    // LIF model parameters
    // **NOTE** copies capacitance from previous benchmark implementation but not 100% sure it's correct
    NeuronModels::LIF::ParamValues lifParams(
        200.0E-9,                       // 0 - C
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

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "EE", SynapseMatrixType::SPARSE_GLOBALG, Parameters::delayTimesteps,
        "E", "E",
        {}, excitatoryStaticSynapseInit,
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::FixedProbabilityNoAutapse>(fixedProb));
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
