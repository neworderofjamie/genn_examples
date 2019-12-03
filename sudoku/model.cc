// GeNN includes
#include "modelSpec.h"

#include "parameters.h"
#include "puzzles.h"

//----------------------------------------------------------------------------
// LIFSpikeCount
//----------------------------------------------------------------------------
class LIFSpikeCount : public NeuronModels::Base
{
public:
    DECLARE_MODEL(LIFSpikeCount, 7, 3);

    SET_PARAM_NAMES({
        "C",            // Membrane capacitance
        "TauM",         // Membrane time constant [ms]
        "Vrest",        // Resting membrane potential [mV]
        "Vreset",       // Reset voltage [mV]
        "Vthresh",      // Spiking threshold [mV]
        "Ioffset",      // Offset current
        "TauRefrac"});  // Refractory time [ms]

    SET_DERIVED_PARAMS({
        {"ExpTC", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[1]); }},
        {"Rmembrane", [](const std::vector<double> &pars, double){ return  pars[1] / pars[0]; }}});

    SET_VARS({{"V", "scalar"}, {"RefracTime", "scalar"}, {"SpikeCount", "unsigned int"}});
    
    SET_SIM_CODE(
        "if ($(RefracTime) <= 0.0) {\n"
        "  scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);\n"
        "  $(V) = alpha - ($(ExpTC) * (alpha - $(V)));\n"
        "}\n"
        "else {\n"
        "  $(RefracTime) -= DT;\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(V) = $(Vreset);\n"
        "$(RefracTime) = $(TauRefrac);\n"
        "$(SpikeCount)++;\n");
    

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(LIFSpikeCount);

//----------------------------------------------------------------------------
// PoissonCurrentSource
//----------------------------------------------------------------------------
class PoissonCurrentSource : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(PoissonCurrentSource, 2, 2);
    
    SET_PARAM_NAMES({
        "Tau",
        "Rate"});
        
    SET_VARS({
        {"Weight", "scalar", VarAccess::READ_ONLY},
        {"Current", "scalar"}});
    
    SET_DERIVED_PARAMS({
        {"ExpDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"Init", [](const std::vector<double> &pars, double dt){ return (1.0 - std::exp(-dt / pars[0])) * (pars[0] / dt); }},
        {"ExpMinusLambda", [](const std::vector<double> &pars, double dt){ return std::exp(-(pars[1] / 1000.0) * dt); }}});
        
    SET_INJECTION_CODE(
        "scalar p = 1.0f;\n"
        "unsigned int numPoissonSpikes = 0;\n"
        "do\n"
        "{\n"
        "    numPoissonSpikes++;\n"
        "    p *= $(gennrand_uniform);\n"
        "} while (p > $(ExpMinusLambda));\n"
        "$(Current) += $(Weight) * $(Init) * (scalar)(numPoissonSpikes - 1);\n"
        "$(injectCurrent, $(Current));\n"
        "$(Current) *= $(ExpDecay);\n");
};
IMPLEMENT_MODEL(PoissonCurrentSource);

//----------------------------------------------------------------------------
// CluePoissonCurrentSource
//----------------------------------------------------------------------------
class CluePoissonCurrentSource : public CurrentSourceModels::Base
{
public:
    DECLARE_MODEL(CluePoissonCurrentSource, 4, 2);
    
    SET_PARAM_NAMES({
        "Tau",
        "Rate",
        "Clue",
        "CoreSize"});
        
    SET_VARS({
        {"Weight", "scalar", VarAccess::READ_ONLY}, 
        {"Current", "scalar"}});
    
    SET_DERIVED_PARAMS({
        {"ExpDecay", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"Init", [](const std::vector<double> &pars, double dt){ return (1.0 - std::exp(-dt / pars[0])) * (pars[0] / dt); }},
        {"ExpMinusLambda", [](const std::vector<double> &pars, double dt){ return std::exp(-(pars[1] / 1000.0) * dt); }}});
        
    SET_INJECTION_CODE(
        "const unsigned int coreSize = (unsigned int)$(CoreSize);\n"
        "const unsigned int clue = ((unsigned int)$(Clue) - 1);\n"
        "if(($(id) / coreSize) == clue) {\n"
        "    scalar p = 1.0f;\n"
        "    unsigned int numPoissonSpikes = 0;\n"
        "    do\n"
        "    {\n"
        "        numPoissonSpikes++;\n"
        "        p *= $(gennrand_uniform);\n"
        "    } while (p > $(ExpMinusLambda));\n"
        "    $(Current) += $(Weight) * $(Init) * (scalar)(numPoissonSpikes - 1);\n"
        "    $(injectCurrent, $(Current));\n"
        "    $(Current) *= $(ExpDecay);\n"
        "}\n");
};
IMPLEMENT_MODEL(CluePoissonCurrentSource);

//----------------------------------------------------------------------------
// DomainToDomain
//----------------------------------------------------------------------------
//! Connects neurons in same domain together
class DomainToDomain : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(DomainToDomain, 1);

    SET_ROW_BUILD_CODE(
        "const unsigned int coreSize = (unsigned int)$(CoreSize);\n"
        "if(c >= coreSize) {\n"
        "   $(endRow);\n"
        "}\n"
        "const unsigned int postDomainStart = coreSize * ($(id_pre) / coreSize);\n"
        "$(addSynapse, postDomainStart + c);\n"
        "c++;\n");

    SET_PARAM_NAMES({"CoreSize"});
    SET_ROW_BUILD_STATE_VARS({{"c", "unsigned int", 0}});
    
    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::vector<double> &pars)
        {
            return (unsigned int)pars[0];
        });
};
IMPLEMENT_SNIPPET(DomainToDomain);

//----------------------------------------------------------------------------
// DomainToNotDomain
//----------------------------------------------------------------------------
//! Connects neurons in one domain to all neurons not in same domain
class DomainToNotDomain : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(DomainToNotDomain, 1);

    SET_ROW_BUILD_CODE(
        "const unsigned int coreSize = (unsigned int)$(CoreSize);\n"
        "const unsigned int notCoreSize = $(num_post) - coreSize;\n"
        "if(c >= notCoreSize) {\n"
        "   $(endRow);\n"
        "}\n"
        "const unsigned int postDomainStart = coreSize * ($(id_pre) / coreSize);\n"
        "if(c < postDomainStart) {\n"
        "    $(addSynapse, c);\n"
        "}\n"
        "else {\n"
        "    $(addSynapse, c + coreSize);\n"
        "}\n"
        "c++;\n");

    SET_PARAM_NAMES({"CoreSize"});
    SET_ROW_BUILD_STATE_VARS({{"c", "unsigned int", 0}});
    
    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int numPost, const std::vector<double> &pars)
        {
            return (numPost - (unsigned int)pars[0]);
        });
};
IMPLEMENT_SNIPPET(DomainToNotDomain);

template<size_t S>
void buildModel(ModelSpec &model, const Puzzle<S> &puzzle) 
{
    const size_t subSize = (size_t)std::sqrt(S);

    InitVarSnippet::Uniform::ParamValues vDist(
        -65.0,  // 0 - min
        -55.0); // 1 - max
    
    // Parameters for LIF neurons
    LIFSpikeCount::ParamValues lifParams(
        0.25,   // Membrane capacitance
        20.0,   // Membrane time constant [ms]
        -65.0,  // Resting membrane potential [mV]
        -70.0,  // Reset voltage [mV]
        -50.0,  // Spiking threshold [mV]
        0.3,    // Offset current [nA]
        2.0);   // Refractory time [ms]

    // Initial values for LIF neurons
    LIFSpikeCount::VarValues lifInit(
        initVar<InitVarSnippet::Uniform>(vDist),    // V
        0.0,                                        // RefracTime
        0);                                         // Spike count
    
    // Parameters for exponentially-shaped synapses
    PostsynapticModels::ExpCurr::ParamValues expCurrParams(
        5.0);   // Tau
    
    // Distribution of weights for noise input
    InitVarSnippet::Uniform::ParamValues stimWeightDist(
        1.4,  // 0 - min
        1.6); // 1 - max
        
    PoissonCurrentSource::ParamValues stimParams(
        5.0,    // Tau [ms]
        20.0);  // Rate [Hz]
    
    PoissonCurrentSource::VarValues stimInitVals(
        initVar<InitVarSnippet::Uniform>(stimWeightDist),   // Weight [nA]
        0.0);                                               // Current [nA]
    
    // Distribution of weights for clue noise input
    InitVarSnippet::Uniform::ParamValues clueWeightDist(
        1.8,  // 0 - min
        2.0); // 1 - max
    
    // Initial values for clue noise input
    CluePoissonCurrentSource::VarValues clueStimInitVals(
        initVar<InitVarSnippet::Uniform>(clueWeightDist),   // Weight
        0.0);                                               // Current [nA]
    
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
            // Create neuron population
            const std::string popName = Parameters::getPopName(x, y);
            auto *neuronPop = model.addNeuronPopulation<LIFSpikeCount>(popName, Parameters::coreSize * 9, lifParams, lifInit);
            neuronPop->setVarLocation("SpikeCount", VarLocation::HOST_DEVICE);
        
            // If this variable state is a clue, add a permanent Poisson current input to strongly excite it
            if(puzzle.puzzle[y][x] != 0) {
                // Parameters for clue noise input
                CluePoissonCurrentSource::ParamValues clueStimParams(
                    5.0,                    // Tau [ms]
                    20.0,                   // Rate [Hz]
                    puzzle.puzzle[y][x],    // Clue
                    Parameters::coreSize);  // Core size
                    
                model.addCurrentSource<CluePoissonCurrentSource>("clue_" + popName, popName, 
                                                                 clueStimParams, clueStimInitVals);
            }
            // Otherwise, add a weaker Poisson current input
            else {
                model.addCurrentSource<PoissonCurrentSource>("stim_" + popName, popName, 
                                                             stimParams, stimInitVals);
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
    
    DomainToNotDomain::ParamValues internalInhibitionParams(Parameters::coreSize);
    
    // Add recurrent inhibition between each variable domain
    for(size_t y = 0; y < S; y++) {
        for(size_t x = 0; x < S; x++) {
            const std::string popName = Parameters::getPopName(x, y);
            model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
                "internalInhibition_" + popName, SynapseMatrixType::SPARSE_INDIVIDUALG, Parameters::delay, popName, popName,
                {}, internalInhibitionInit,
                expCurrParams, {},
                initConnectivity<DomainToNotDomain>(internalInhibitionParams));
        }
    }
    
    // sudoku.apply_constraints(w_range=[-0.2/2.5, 0.0])*/
    // Uniformly distribute weights
    InitVarSnippet::Uniform::ParamValues constraintGDist(
        -0.2 / 2.5, // 0 - min
        0.0);       // 1 - max
    
    WeightUpdateModels::StaticPulse::VarValues constraintInit(
        initVar<InitVarSnippet::Uniform>(constraintGDist)); // g
    
    DomainToDomain::ParamValues constraintParams(Parameters::coreSize);
    size_t pre = 0;
    for(size_t yPre = 0; yPre < S; yPre++) {
        for(size_t xPre = 0; xPre < S; xPre++) {
            const std::string preName = Parameters::getPopName(xPre, yPre);
            size_t post = 0;
            for(size_t yPost = 0; yPost < S; yPost++) {
                for(size_t xPost = 0; xPost < S; xPost++) {
                    const std::string postName = Parameters::getPopName(xPost, yPost);
                    
                    // **TODO** more elegant way of achieving triangle
                    if(post > pre) {
                        // If there should be a horizontal or vertical constraint
                        if((xPre == xPost || yPre == yPost)) {
                            model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
                                "lineConstraint_" + preName + "_" + postName, SynapseMatrixType::SPARSE_INDIVIDUALG, Parameters::delay, preName, postName,
                                {}, constraintInit,
                                expCurrParams, {},
                                initConnectivity<DomainToDomain>(constraintParams));
                        }
                        
                        // If variables are in same 3X3 square & (different row & different column)
                        if(((xPre / subSize) == (xPost / subSize)) && ((yPre / subSize) == (yPost / subSize))
                            && (xPre != xPost) && (yPre != yPost))
                        {
                            model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::ExpCurr>(
                                "subConstraint_" + preName + "_" + postName, SynapseMatrixType::SPARSE_INDIVIDUALG, Parameters::delay, preName, postName,
                                {}, constraintInit,
                                expCurrParams, {},
                                initConnectivity<DomainToDomain>(constraintParams));
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
    model.setName("sudoku");
    model.setMergePostsynapticModels(true);
    model.setDefaultVarLocation(VarLocation::DEVICE);
    model.setDefaultSparseConnectivityLocation(VarLocation::DEVICE);
    
    buildModel(model, Puzzles::easy);
}