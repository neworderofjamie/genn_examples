#include <cmath>
#include <limits>
#include <vector>

// GeNN includes
#include "modelSpec.h"

// GeNN robotics includes
#include "connectors.h"
#include "exp_curr.h"

// Examples common includes
#include "../common/bcpnn.h"

// Model includes
#include "parameters.h"

//----------------------------------------------------------------------------
// LIFPoisson
//----------------------------------------------------------------------------
//! Leaky integrate-and-fire neuron solved algebraically
class LIFPoisson : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFPoisson, 11, 3);

    SET_SIM_CODE(
        "const scalar stimPoissonExpMinusLambda = *($(stimPoissonExpMinusLambda) + ($(id) / 100));\n"
        "scalar pStim = 1.0f;\n"
        "unsigned int numStimPoissonSpikes = 0;\n"
        "do\n"
        "{\n"
        "    numStimPoissonSpikes++;\n"
        "    pStim *= $(gennrand_uniform);\n"
        "} while (pStim > stimPoissonExpMinusLambda);\n"
        "scalar pBackground = 1.0f;\n"
        "unsigned int numBackgroundPoissonSpikes = 0;\n"
        "do\n"
        "{\n"
        "    numBackgroundPoissonSpikes++;\n"
        "    pBackground *= $(gennrand_uniform);\n"
        "} while (pBackground > $(BackgroundPoissonExpMinusLambda));\n"
        "$(Ipoisson) += $(PoissonInit) * (($(BackgroundPoissonWeight) * (scalar)(numBackgroundPoissonSpikes - 1)) + ($(StimPoissonWeight) * (scalar)(numStimPoissonSpikes - 1)));\n"
        "if ($(RefracTime) <= 0.0)\n"
        "{\n"
        "  scalar alpha = (($(Isyn) + $(Ioffset) + $(Ipoisson)) * $(Rmembrane)) + $(Vrest);\n"
        "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else\n"
        "{\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
        "$(Ipoisson) *= $(PoissonExpDecay);\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "C",                        // 0 - Membrane capacitance
        "TauM",                     // 1 - Membrane time constant [ms]
        "Vrest",                    // 2 - Resting membrane potential [mV]
        "Vreset",                   // 3 - Reset voltage [mV]
        "Vthresh",                  // 4 - Spiking threshold [mV]
        "Ioffset",                  // 5 - Offset current
        "TauRefrac",                // 6 - Refractory time [ms]
        "BackgroundPoissonRate",    // 7 - Poisson input rate [Hz]
        "BackgroundPoissonWeight",  // 8 - How much current each poisson spike adds [nA]
        "StimPoissonWeight",        // 9 - How much current each poisson spike adds [nA]
        "PoissonTau"});             // 10 - Time constant of poisson spike integration [ms]


    SET_DERIVED_PARAMS({
        {"ExpTC", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const vector<double> &pars, double){ return  pars[1] / pars[0]; }},
        {"BackgroundPoissonExpMinusLambda", [](const vector<double> &pars, double dt){ return std::exp(-(pars[7] / 1000.0) * dt); }},
        {"PoissonExpDecay", [](const vector<double> &pars, double dt){ return std::exp(-dt / pars[10]); }},
        {"PoissonInit", [](const vector<double> &pars, double dt){ return (1.0 - std::exp(-dt / pars[10])) * (pars[10] / dt); }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"Ipoisson", "scalar"}});

    SET_EXTRA_GLOBAL_PARAMS({
        {"stimPoissonExpMinusLambda", "scalar*"},
    });
};
IMPLEMENT_MODEL(LIFPoisson);

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(1.0);
    model.setName("bcpnn_modular");

    GENN_PREFERENCES::buildSharedLibrary = true;
    GENN_PREFERENCES::optimizeCode = true;
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultSparseConnectivityMode = VarMode::LOC_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    InitSparseConnectivitySnippet::FixedProbability::ParamValues fixedProb(
        Parameters::probabilityConnection); // 0 - prob

    InitVarSnippet::Uniform::ParamValues vDist(
        -80.0,                                  // 0 - min
        Parameters::neuronThresholdVoltage);    // 1 - max

    // LIF model parameters
    LIFPoisson::ParamValues lifParams(
        0.25,                                   // 0 - C
        Parameters::neuronTauMem,               // 1 - TauM
        Parameters::neuronResetVoltage,         // 2 - Vrest
        Parameters::neuronResetVoltage,         // 3 - Vreset
        Parameters::neuronThresholdVoltage,     // 4 - Vthresh
        0.0,                                    // 5 - Ioffset
        2.0,                                    // 6 - TauRefrac
        Parameters::backgroundRate,             // 7 - Background poisson input rate
        Parameters::backgroundWeightTraining,   // 8 - Background poisson input weight
        Parameters::stimWeightTraining,         // 9 - Stimuli poisson input rate
        Parameters::tauSynAMPAGABA);            // 10 - Time constant of Poisson input integration

    // LIF initial conditions
    LIFPoisson::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),    // 0 - V
        0.0,                                        // 1 - RefracTime
        0.0);                                       // 2 - IPoisson

    // BCPNN params
    BCPNNTwoTrace::ParamValues bcpnnAMPAParams(
        Parameters::tauZiAMPA,      // 0 - Time constant of presynaptic primary trace (ms)
        Parameters::tauZjAMPA,      // 1 - Time constant of postsynaptic primary trace (ms)
        Parameters::tauP,           // 2 - Time constant of probability trace
        Parameters::fmax,           // 3 - Maximum firing frequency (Hz)
        1.0,                        // 4 - Maximum weight
        false,                      // 5 - Should weights get applied to synapses
        true);                      // 6 - Should weights be updated

    BCPNNTwoTrace::ParamValues bcpnnNMDAParams(
        Parameters::tauZiNMDA,      // 0 - Time constant of presynaptic primary trace (ms)
        Parameters::tauZjNMDA,      // 1 - Time constant of postsynaptic primary trace (ms)
        Parameters::tauP,           // 2 - Time constant of probability trace
        Parameters::fmax,           // 3 - Maximum firing frequency (Hz)
        1.0,                        // 4 - Maximum weight
        false,                      // 5 - Should weights get applied to synapses
        true);                      // 6 - Should weights be updated

    BCPNNTwoTrace::VarValues bcpnnInit(
        0.0,                                    // 0 - g
        0.0,                                    // 1 - PijStar
        std::numeric_limits<float>::lowest());  // 2 - lastUpdateTime

    BCPNNTwoTrace::PreVarValues bcpnnPreInit(
        0.0,    // 0 - ZiStar
        0.0);   // 1 - PiStar

    BCPNNTwoTrace::PostVarValues bcpnnPostInit(
        0.0,    // 0 - ZjStar
        0.0);   // 1 - PjStar

    // Exponential current parameters
    ExpCurr::ParamValues ampaGABAExpCurrParams(
        Parameters::tauSynAMPAGABA);  // 0 - TauSyn (ms)

    ExpCurr::ParamValues nmdaExpCurrParams(
        Parameters::tauSynNMDA);  // 0 - TauSyn (ms)

    // Loop through hypercolumns
    for(unsigned int i = 0; i < Parameters::numHC; i++) {
        const std::string name = "E_" + std::to_string(i);

        // Create population of excutatory neurons for hypercolumn
        auto *e = model.addNeuronPopulation<LIFPoisson>(name, Parameters::numHCExcitatoryNeurons, lifParams, lifInit);
       
        // Configure spike variables so that they can be downloaded to host
        e->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
    }

    // Loop through connections between hypercolumns
    for(unsigned int i = 0; i < Parameters::numHC; i++) {
        const std::string preName = "E_" + std::to_string(i);
        for(unsigned int j = 0; j < Parameters::numHC; j++) {
            const std::string postName = "E_" + std::to_string(j);

            // Create AMPA and NMDA connections between hypercolumns
            const std::string synapseName = preName + "_" + postName;
            auto *eeAMPA = model.addSynapsePopulation<BCPNNTwoTrace, ExpCurr>(
                synapseName + "_AMPA", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
                preName, postName,
                bcpnnAMPAParams, bcpnnInit, bcpnnPreInit, bcpnnPostInit,
                ampaGABAExpCurrParams, {},
                initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

            auto *eeNMDA = model.addSynapsePopulation<BCPNNTwoTrace, ExpCurr>(
                synapseName + "_NMDA", SynapseMatrixType::RAGGED_INDIVIDUALG, NO_DELAY,
                preName, postName,
                bcpnnNMDAParams, bcpnnInit, bcpnnPreInit, bcpnnPostInit,
                nmdaExpCurrParams, {},
                initConnectivity<InitSparseConnectivitySnippet::FixedProbability>(fixedProb));

            // Configure synaptic weights and connectivity so they are allocated on host and device, allowing them to be downloaded
            eeAMPA->setWUVarMode("g", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
            eeNMDA->setWUVarMode("g", VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
            eeAMPA->setSparseConnectivityVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);
            eeNMDA->setSparseConnectivityVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

            // Set max connections
            const unsigned int maxConnections = calcFixedProbabilityConnectorMaxConnections(Parameters::numHCExcitatoryNeurons, Parameters::numHCExcitatoryNeurons,
                                                                                            Parameters::probabilityConnection);
            eeAMPA->setMaxConnections(maxConnections);
            eeNMDA->setMaxConnections(maxConnections);
        }
    }

    model.finalize();
}