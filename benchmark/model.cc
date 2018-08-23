#include <cmath>
#include <vector>

#include "modelSpec.h"

// GeNN robotics includes
#include "genn_utils/connectors.h"
#include "genn_models/exp_curr.h"
#include "genn_models/lif.h"

#include "parameters.h"

using namespace BoBRobotics;

//----------------------------------------------------------------------------
// InitSparseConnectivitySnippet::FixedProbability
//----------------------------------------------------------------------------
//! Initialises variable by sampling from the uniform distribution
class FixedNumberTotalWithReplacement : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(FixedNumberTotalWithReplacement, 2);

    SET_ROW_BUILD_CODE(
        "const scalar u = $(gennrand_uniform);\n"
        "const unsigned int postIdx = ceil(u * (scalar)$(numPost)) - 1;\n"
        "$(addSynapse, postIdx);\n"
        "$(prevJ)++;\n"
        "if($(prevJ) >= ($(rowLength)[$(id_pre)] - 1)) {\n"
        "   $(endRow);\n"
        "}\n");

    SET_PARAM_NAMES({"total", "numPost"});
    SET_EXTRA_GLOBAL_PARAMS({{"rowLength", "unsigned int*"}})

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected, and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, pars[0], (double)numPost / ((double)numPre * (double)numPost));
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPost);

            // There are numConnections connections amongst the numPre*numPost possible connections.
            // Each of the numConnections connections has an independent p=float(numPost)/(numPre*numPost)
            // probability of being selected, and the number of synapses in the sub-row is binomially distributed
            return binomialInverseCDF(quantile, pars[0], (double)numPre / ((double)numPre * (double)numPost));
        });

    SET_NEEDS_ROW_SORT(true);
};
IMPLEMENT_SNIPPET(FixedNumberTotalWithReplacement);

void modelDefinition(NNmodel &model)
{
    // Enable new automatic initialisation mode
    GENN_PREFERENCES::autoInitSparseVars = true;
    GENN_PREFERENCES::defaultVarMode = VarMode::LOC_DEVICE_INIT_DEVICE;
    GENN_PREFERENCES::defaultSparseConnectivityMode = VarMode::LOC_HOST_DEVICE_INIT_DEVICE;

    initGeNN();
    model.setDT(1.0);
    model.setName("benchmark");
    model.setTiming(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    GeNNModels::LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.0,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    // **TODO** uniform random
    GeNNModels::LIF::VarValues lifInit(
        -55.0,  // 0 - V
        0.0);    // 1 - RefracTime

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        0.1);    // 0 - Wij (nA)

    // Exponential current parameters
    GeNNModels::ExpCurr::ParamValues expCurrParams(
        5.0);  // 0 - TauSyn (ms)

    FixedNumberTotalWithReplacement::ParamValues connectParams(Parameters::numConnections,
                                                               Parameters::numNeurons);

    // Create IF_curr neuron
    model.addNeuronPopulation<GeNNModels::LIF>("Neurons", Parameters::numNeurons,
                                               lifParams, lifInit);

    auto *syn = model.addSynapsePopulation<WeightUpdateModels::StaticPulse, GeNNModels::ExpCurr>("Syn", SYNAPSE_MATRIX_TYPE, NO_DELAY,
                                                                                                 "Neurons", "Neurons",
                                                                                                 {}, staticSynapseInit,
                                                                                                 expCurrParams, {},
                                                                                                 initConnectivity<FixedNumberTotalWithReplacement>(connectParams));

    model.finalize();
}