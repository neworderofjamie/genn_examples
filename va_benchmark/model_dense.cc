#include <fstream>

#include "modelSpec.h"

#include "parameters.h"


class LIFHalf : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(LIFHalf);

    SET_SIM_CODE(
        "if (RefracTime <= 0.0) {\n"
        "  scalar alpha = ((Isyn + Ioffset) * Rmembrane) + Vrest;\n"
        "  V = alpha - (ExpTC * (alpha - V));\n"
        "}\n"
        "else {\n"
        "  RefracTime -= dt;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("RefracTime <= 0.0 && V >= Vthresh");

    SET_RESET_CODE(
        "V = Vreset;\n"
        "RefracTime = TauRefrac;\n");

    SET_PARAMS({
        "C",          // Membrane capacitance
        "TauM",       // Membrane time constant [ms]
        "Vrest",      // Resting membrane potential [mV]
        "Vreset",     // Reset voltage [mV]
        "Vthresh",    // Spiking threshold [mV]
        "Ioffset",    // Offset current
        "TauRefrac"});

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauM").cast<double>()); }},
        {"Rmembrane", [](const ParamValues &pars, double){ return  pars.at("TauM").cast<double>() / pars.at("C").cast<double>(); }}});

    SET_VARS({{"V", "scalar", "half"}, {"RefracTime", "scalar", "half"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_SNIPPET(LIFHalf);

class StaticPulseHalf : public WeightUpdateModels::Base
{
public:
    DECLARE_SNIPPET(StaticPulseHalf);

    SET_VARS({{"g", "scalar", "half", VarAccess::READ_ONLY}});

    SET_PRE_SPIKE_SYN_CODE("addToPost(g);\n");
};
IMPLEMENT_SNIPPET(StaticPulseHalf);

class FixedProbDense : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(FixedProbDense);

    SET_CODE("value = (gennrand_uniform() < prob) ? weight : 0.0;");

    SET_PARAMS({"prob", "weight"});
};
IMPLEMENT_SNIPPET(FixedProbDense);

void modelDefinition(ModelSpec &model)
{
    //GENN_PREFERENCES.debugCode = true;
    GENN_PREFERENCES.generateLineInfo = true;

    model.setDT(1.0);
    model.setName("va_benchmark_dense");
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    model.setTimingEnabled(true);
    model.setDefaultNarrowSparseIndEnabled(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    ParamValues vDist{
        {"min", Parameters::resetVoltage},
        {"max", Parameters::thresholdVoltage}};

    ParamValues fixedProb{{"prob", Parameters::probabilityConnection}};

    // LIF model parameters
    ParamValues lifParams{
        {"C", 1.0},
        {"TauM", 20.0},
        {"Vrest", -49.0},
        {"Vreset", Parameters::resetVoltage},
        {"Vthresh", Parameters::thresholdVoltage},
        {"Ioffset", 0.0},
        {"TauRefrac", 5.0}};

    // LIF initial conditions
    VarValues lifInit{
        {"V", initVar<InitVarSnippet::Uniform>(vDist)},
        {"RefracTime", 0.0}};
    
    ParamValues fixedProbExcInit{
        {"prob", Parameters::probabilityConnection},
        {"weight", Parameters::excitatoryWeight}};
        
    ParamValues fixedProbInhInit{
        {"prob", Parameters::probabilityConnection},
        {"weight", Parameters::inhibitoryWeight}};
        
    
    // Static synapse parameters
    VarValues excitatoryStaticSynapseInit{{"g", initVar<FixedProbDense>(fixedProbExcInit)}};
    VarValues inhibitoryStaticSynapseInit{{"g", initVar<FixedProbDense>(fixedProbInhInit)}};

    // Exponential current parameters
    ParamValues excitatoryExpCurrParams{{"tau", 5.0}};
    ParamValues inhibitoryExpCurrParams{{"tau", 10.0}};

    // Create IF_curr neuron
    auto *e = model.addNeuronPopulation<LIFHalf>("E", Parameters::numExcitatory, lifParams, lifInit);
    auto *i = model.addNeuronPopulation<LIFHalf>("I", Parameters::numInhibitory, lifParams, lifInit);

    // Enable spike recording
    e->setSpikeRecordingEnabled(true);
    i->setSpikeRecordingEnabled(true);

    // Determine matrix type
    const SynapseMatrixType matrixType = SynapseMatrixType::DENSE;
    
    using WUM = StaticPulseHalf;
    auto *ee = model.addSynapsePopulation(
        "EE", matrixType, e, e,
        initWeightUpdate<WUM>({}, excitatoryStaticSynapseInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(excitatoryExpCurrParams));
    auto *ei = model.addSynapsePopulation(
        "EI", matrixType, e, i,
        initWeightUpdate<WUM>({}, excitatoryStaticSynapseInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(excitatoryExpCurrParams));
    auto *ii = model.addSynapsePopulation(
        "II", matrixType, i, i,
        initWeightUpdate<WUM>({}, inhibitoryStaticSynapseInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(inhibitoryExpCurrParams));
    auto *ie = model.addSynapsePopulation(
        "IE", matrixType, i, e,
        initWeightUpdate<WUM>({}, inhibitoryStaticSynapseInit),
        initPostsynaptic<PostsynapticModels::ExpCurr>(inhibitoryExpCurrParams));
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
    runtime.allocate(Parameters::numTimesteps);
    runtime.initialize();
    runtime.initializeSparse();
    
    const auto startTime = std::chrono::high_resolution_clock::now();
    while(runtime.getTimestep() < Parameters::numTimesteps) {
        runtime.stepTime();
    }
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - startTime;
    std::cout << "Init time:" << runtime.getInitTime() << std::endl;
    std::cout << "Total simulation time:" << duration.count() << " seconds" << std::endl;
    std::cout << "\tNeuron update time:" << runtime.getNeuronUpdateTime() << std::endl;
    std::cout << "\tPresynaptic update time:" << runtime.getPresynapticUpdateTime() << std::endl;
    
    runtime.pullRecordingBuffersFromDevice();
    
    const auto *e = model.findNeuronGroup("E");
    const auto *i = model.findNeuronGroup("I");
    runtime.writeRecordedSpikes(*e, "spikes_e.csv");
    runtime.writeRecordedSpikes(*i, "spikes_i.csv");

}

