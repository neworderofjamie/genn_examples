// GeNN includes
#include "modelSpec.h"

#include "parameters.h"
#include "puzzles.h"

//----------------------------------------------------------------------------
// LIFPoisson
//----------------------------------------------------------------------------
//! Leaky integrate-and-fire neuron solved algebraically
class LIFPoisson : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFPoisson, 9, 3);

    SET_SIM_CODE(
        "scalar p = 1.0f;\n"
        "unsigned int numPoissonSpikes = 0;\n"
        "do\n"
        "{\n"
        "    numPoissonSpikes++;\n"
        "    p *= $(gennrand_uniform);\n"
        "} while (p > $(PoissonExpMinusLambda));\n"
        "$(Ipoisson) += $(IpoissonInit) * (scalar)(numPoissonSpikes - 1);\n"
        "if ($(RefracTime) <= 0.0)\n"
        "{\n"
        "  scalar alpha = (($(Isyn) + $(Ioffset) + $(Ipoisson)) * $(Rmembrane)) + $(Vrest);\n"
        "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else\n"
        "{\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
        "$(Ipoisson) *= $(IpoissonExpDecay);\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n");

    SET_PARAM_NAMES({
        "C",                // Membrane capacitance
        "TauM",             // Membrane time constant [ms]
        "Vrest",            // Resting membrane potential [mV]
        "Vreset",           // Reset voltage [mV]
        "Vthresh",          // Spiking threshold [mV]
        "TauRefrac",        // Refractory time [ms]
        "PoissonWeight",    // How much current each poisson spike adds [nA]
        "IpoissonTau",      // Time constant of poisson spike integration [ms]
        "Ioffset"});        // Offset current

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }},
        {"IpoissonExpDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[7]); }},
        {"IpoissonInit", [](const std::vector<double> &pars, double dt){ return pars[6] * (1.0 - std::exp(-dt / pars[7])) * (pars[7] / dt); }}});

    SET_EXTRA_GLOBAL_PARAMS({{"PoissonExpMinusLambda",  "scalar"}});    // Lambda for Poisson process

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"Ipoisson", "scalar"}});
};
IMPLEMENT_MODEL(LIFPoisson);

std::string getPopName(size_t x, size_t y, size_t d)
{
    return std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(d);
}

template<size_t S>
void buildModel(ModelSpec &model, const Puzzle<S> &puzzle) 
{
    const size_t subSize = (size_t)std::sqrt(S);

    InitVarSnippet::Uniform::ParamValues vDist(
        -65.0,  // 0 - min
        -55.0); // 1 - max
    
    // Parameters for LIF neurons
    LIFPoisson::ParamValues lifParams(
        0.25,   // Membrane capacitance
        20.0,   // Membrane time constant [ms]
        -65.0,  // Resting membrane potential [mV]
        -70.0,  // Reset voltage [mV]
        -50.0,  // Spiking threshold [mV]
        2.0,    // Refractory time [ms]
        0.0,    // How much current each poisson spike adds [nA]
        10.0,      // Time constant of poisson spike integration [ms]
        0.3);
    
    // Initial values for LIF neurons
    LIFPoisson::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),    // V
        0.0,                                        // RefracTime
        0.0);                                       // Ipoisson);   // V
    
    // Parameters for exponentially-shaped synapses
    PostsynapticModels::ExpCurr::ParamValues expCurrParams(
        5.0);   // Tau
        
    // Add neuron populations for each variable state
    //sudoku.build_domains_pops()
    // sudoku.build_stimulation_pops(1, shrink=1.0,stim_ratio=1.,rate=(20.0, 20.0),full=True, phase=0.0, clue_size=None))
    // > poisson for each neuron (random start and stop)
    // > poisson for each clue
    // sudoku.build_dissipation_pops(d_populations=1, shrink=1.0, stim_ratio=1.0, rate=20.0, full=True, phase=0.0)
    // > dissipative poisson for each neuron (random start and stop) - SEEMS disabled
    // sudoku.stimulate_cores(w_range=[1.4, 1.6], d_range=[1.0, 1.0], w_clues=[1.8, 2.0]) # , w_clues=[1.5, 1.9])
    
    for(size_t y = 0; y < S; y++) {
        for(size_t x = 0; x < S; x++) {
            for(size_t d = 1; d < 10; d++) {
                model.addNeuronPopulation<LIFPoisson>(getPopName(x, y, d), Parameters::coreSize, lifParams, lifInit);
                
                // If this variable state is a clue
                if(puzzle.puzzle[y][x] == d) {
                }
                
                
            }
            
            
        }
    }
    
    // sudoku.internal_inhibition(w_range=[-0.2/2.5, 0.0])
    
    // Uniformly distribute weights
    InitVarSnippet::Uniform::ParamValues internalInhibitionGDist(
        -0.2 / 2.5, // 0 - min
        0.0);       // 1 - max
    
    WeightUpdateModels::StaticPulse::VarValues internalInhibitionInit(
        initVar<InitVarSnippet::Uniform>(internalInhibitionGDist)); // g
        
    // Connect all domain populations for a variable together
    for(size_t y = 0; y < S; y++) {
        for(size_t x = 0; x < S; x++) {
            for(size_t dPre = 1; dPre < 10; dPre++) {
                const std::string preName = getPopName(x, y, dPre);
                for(size_t dPost = 1; dPost < 10; dPost++) {
                    const std::string postName = getPopName(x, y, dPost);
                    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
                        "internalInhibition_" + preName + "_" + postName, SynapseMatrixType::DENSE_INDIVIDUALG, Parameters::delay, preName, postName,
                        {}, internalInhibitionInit,
                        expCurrParams, {});
                }
            }
        }
    }
    
    // sudoku.apply_constraints(w_range=[-0.2/2.5, 0.0])*/
    
    // Uniformly distribute weights
    InitVarSnippet::Uniform::ParamValues constraintGDist(
        -0.2 / 2.5, // 0 - min
        0.0);       // 1 - max
    
    WeightUpdateModels::StaticPulse::VarValues constraintInit(
        initVar<InitVarSnippet::Uniform>(constraintGDist)); // g
    
    size_t pre = 0;
    for(size_t yPre = 0; yPre < S; yPre++) {
        for(size_t xPre = 0; xPre < S; xPre++) {
            size_t post = 0;
            for(size_t yPost = 0; yPost < S; yPost++) {
                for(size_t xPost = 0; xPost < S; xPost++) {
                    // **TODO** more elegant way of achieving triangle
                    if(post > pre) {
                        // If there should be a horizontal or vertical constraint
                        if((xPre == xPost || yPre == yPost)) {
                            for(size_t d = 1; d < 10; d++) {
                                const std::string preName = getPopName(xPre, yPre, d);
                                const std::string postName = getPopName(xPost, yPost, d);
                                model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
                                    "lineConstraint_" + preName + "_" + postName, SynapseMatrixType::DENSE_INDIVIDUALG, Parameters::delay, preName, postName,
                                    {}, constraintInit,
                                    expCurrParams, {});
                            }
                        }
                        
                        // If variables are in same 3X3 square & (different row & different column)
                        if(((xPre / subSize) == (xPost / subSize)) && ((yPre / subSize) == (yPost / subSize))
                            && (xPre != xPost) && (yPre != yPost))
                        {
                            for(size_t d = 1; d < 10; d++) {
                                const std::string preName = getPopName(xPre, yPre, d);
                                const std::string postName = getPopName(xPost, yPost, d);
                                model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
                                    "subConstraint_" + preName + "_" + postName, SynapseMatrixType::DENSE_INDIVIDUALG, Parameters::delay, preName, postName,
                                    {}, constraintInit,
                                    expCurrParams, {});
                            }
                        }
                    }
                    post++;
                }
            }
            
            pre++;
        }
    }
}

void modelDefinition(ModelSpec &model)
{
    model.setDT(1.0);
    model.setName("potjans_sudoku");
    model.setMergePostsynapticModels(true);
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    
    buildModel(model, Puzzles::easy);
}