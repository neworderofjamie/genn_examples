
// GeNN includes
#include "modelSpec.h"

// GeNN robotics includes
#include "connectors.h"
#include "exp_curr.h"
#include "lif.h"

// Model includes
#include "parameters.h"

void modelDefinition(NNmodel &model)
{
    initGeNN();
    model.setDT(0.1);
    model.setName("potjans_microcircuit");

    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;

    InitVarSnippet::Normal::ParamValues vDist(
        -58.0, // 0 - mean
        5.0);  // 1 - sd

    // LIF model parameters
    LIF::ParamValues lifParams(
        0.25,    // 0 - C
        10.0,   // 1 - TauM
        -65.0,  // 2 - Vrest
        -65.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        2.0);    // 6 - TauRefrac

    // LIF initial conditions
    LIF::VarValues lifInit(
        initVar<InitVarSnippet::Normal>(vDist),     // 0 - V
        0.0);   // 1 - RefracTime

    NeuronModels::PoissonNew::VarValues poissonInit(
       0.0);     // 2 - SpikeTime

    // Weight for external input synapses
    WeightUpdateModels::StaticPulse::VarValues extStaticSynapseInit(
        Parameters::externalW);    // 0 - Wij (nA)

    // Exponential current parameters
    ExpCurr::ParamValues excitatoryExpCurrParams(
        0.5);  // 0 - TauSyn (ms)

    ExpCurr::ParamValues inhibitoryExpCurrParams(
        0.5);  // 0 - TauSyn (ms)

    // Loop through populations and layers
    for(unsigned int layer = 0; layer < Parameters::LayerMax; layer++) {
        for(unsigned int pop = 0; pop < Parameters::PopulationMax; pop++) {
            // Determine name of population
            const std::string popName = Parameters::getPopulationName(layer, pop);

            // Create population
            auto *neuronPop = model.addNeuronPopulation<LIF>(popName,
                                                             Parameters::numNeurons[layer][pop],
                                                             lifParams, lifInit);

            // Make recordable on host
            neuronPop->setSpikeVarMode(VarMode::LOC_HOST_DEVICE_INIT_DEVICE);

            // Calculate poisson input rate
            NeuronModels::PoissonNew::ParamValues poissonParams(
                Parameters::externalInputDegrees[layer][pop] * Parameters::backgroundRate);     // 0 - Input rate

            // Create poisson input source
            model.addNeuronPopulation<NeuronModels::PoissonNew>(popName + "input",
                                                                Parameters::numNeurons[layer][pop],
                                                                poissonParams, poissonInit);
        }
    }

    // Loop through target populations and layers
    for(unsigned int trgLayer = 0; trgLayer < Parameters::LayerMax; trgLayer++) {
        for(unsigned int trgPop = 0; trgPop < Parameters::PopulationMax; trgPop++) {
            // Read target population size
            const unsigned numTrg = Parameters::numNeurons[trgLayer][trgPop];
            const std::string trgName = Parameters::getPopulationName(trgLayer, trgPop);

            for(unsigned int srcLayer = 0; srcLayer < Parameters::LayerMax; srcLayer++) {
                for(unsigned int srcPop = 0; srcPop < Parameters::PopulationMax; srcPop++) {
                    // Read source population size
                    const unsigned numSrc = Parameters::numNeurons[trgLayer][trgPop];
                    const std::string srcName = Parameters::getPopulationName(srcLayer, srcPop);

                    const double connectionProb = Parameters::connectionProbabilities[(trgLayer * 2) + trgPop][(srcLayer * 2) + srcPop];
                    const unsigned int k = round(log(1.0 - connectionProb) / log((double)(numTrg * numSrc - 1) / (double)(numTrg * numSrc))) / (double)numTrg;
             /*
              *
                    K[target_index][source_index] = round(np.log(1. - conn_probs[target_index][source_index]) / np.log((n_target * n_source - 1.) / (n_target * n_source))) / n_target*/

                    // Determine mean weight
                    float meanWeight;
                    if(srcPop == Parameters::PopulationE) {
                        if(srcLayer == Parameters::Layer4 && trgLayer == Parameters::Layer23 && trgPop == Parameters::PopulationE) {
                            meanWeight = Parameters::layer234W;
                        }
                        else {
                            meanWeight = Parameters::meanW;
                        }
                    }
                    else {
                        meanWeight = Parameters::g * Parameters::meanW;
                    }

                    // Determine weight standard deviation
                    double weightSD;
                    if(srcPop == Parameters::PopulationE && srcLayer == Parameters::Layer4 && trgLayer == Parameters::Layer23 && trgPop == Parameters::PopulationE) {
                        weightSD = meanWeight * Parameters::layer234RelW;
                    }
                    else {
                        weightSD = fabs(meanWeight * Parameters::relW);
                    }

                    // If there are any connections
                    if(k > 0) {
                        std::cout << "Connection between '" << srcName << "' and '" << trgName << "': K=" << k << ", meanWeight=" << meanWeight << ", weightSD=" << weightSD << std::endl;
                    }
                }
            }
        }
    }

    // Finalise model
    model.finalize();
}