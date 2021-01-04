#include "modelSpec.h"

#include "parameters.h"

class Conv2DSparse : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(Conv2DSparse, 12);

    SET_PARAM_NAMES({"conv_kh", "conv_kw",
                     "conv_sh", "conv_sw",
                     "conv_padh", "conv_padw",
                     "conv_ih", "conv_iw", "conv_ic",
                     "conv_oh", "conv_ow", "conv_oc"});

    SET_ROW_BUILD_STATE_VARS({{"inRow", "int", "($(id_pre) / (int)$(conv_ic)) / (int)$(conv_iw)"},
                              {"inCol", "int", "($(id_pre) / (int)$(conv_ic)) % (int)$(conv_iw)"},
                              {"inChan", "int", "$(id_pre) % (int)$(conv_ic)"},
                              {"outRow", "int", "min((int)$(conv_oh), max(0, 1 + ((inRow + (int)$(conv_padh) - (int)$(conv_kh)) / (int)$(conv_sh))))"},
                              {"maxOutRow", "int", "min((int)$(conv_oh), max(0, 1 + ((inRow + (int)$(conv_padh)) / (int)$(conv_sh))))"},
                              {"minOutCol", "int", "min((int)$(conv_ow), max(0, 1 + ((inCol + (int)$(conv_padw) - (int)$(conv_kw)) / (int)$(conv_sw))))"},
                              {"maxOutCol", "int", "min((int)$(conv_ow), max(0, 1 + ((inCol + (int)$(conv_padw)) / (int)$(conv_sw))))"}});

    SET_ROW_BUILD_CODE(
        "if($(outRow) == $(maxOutRow)) {\n"
        "   $(endRow);\n"
        "}\n"
        "const int strideRow = ($(outRow) * (int)$(conv_sh)) - (int)$(conv_padh);\n"
        "const int kernRow = $(inRow) - strideRow;\n"
        "for(int outCol = $(minOutCol); outCol < $(maxOutCol); outCol++) {\n"
        "    const int strideCol = (outCol * (int)$(conv_sw)) - (int)$(conv_padw);\n"
        "    const int kernCol = $(inCol) - strideCol;\n"
        "    for(unsigned int outChan = 0; outChan < (unsigned int)$(conv_oc); outChan++) {\n"
        "        const int idPost = (($(outRow) * (int)$(conv_ow) * (int)$(conv_oc)) +\n"
        "                           (outCol * (int)$(conv_oc)) +\n"
        "                           outChan);\n"
        "        $(addSynapse, idPost, kernRow, kernCol, $(inChan), outChan);\n"
        "    }\n"
        "}\n"
        "$(outRow)++;\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::vector<double> &pars)
        {
            const unsigned int conv_kh = (unsigned int)pars[0];
            const unsigned int conv_kw = (unsigned int)pars[1];
            const unsigned int conv_sh = (unsigned int)pars[2];
            const unsigned int conv_sw = (unsigned int)pars[3];
            const unsigned int conv_oc = (unsigned int)pars[11];
            return (conv_kh / conv_sh) * (conv_kw / conv_sw) * conv_oc;
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const std::vector<double> &pars)->std::vector<unsigned int>
        {
            return {(unsigned int)pars[0], (unsigned int)pars[1],
                    (unsigned int)pars[8], (unsigned int)pars[11]};
        });
};
IMPLEMENT_SNIPPET(Conv2DSparse);

void modelDefinition(NNmodel &model)
{
    GENN_PREFERENCES.generateLineInfo = true;
    model.setDT(1.0);
    model.setName("benchmark");
    model.setTiming(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters
    NeuronModels::LIF::ParamValues lifParams(
        0.2,    // 0 - C
        20.0,   // 1 - TauM
        -60.0,  // 2 - Vrest
        -60.0,  // 3 - Vreset
        -50.0,  // 4 - Vthresh
        0.5,    // 5 - Ioffset
        5.0);    // 6 - TauRefrac

    // LIF initial conditions
    NeuronModels::LIF::VarValues lifInit(
        -55.0,  // 0 - V
        0.0);    // 1 - RefracTime

    NeuronModels::PoissonNew::ParamValues poissonParams(
        Parameters::inputRate);  // 0 - rate [hz]

    NeuronModels::PoissonNew::VarValues poissonInit(
        0.0);   // 0 - time to spike [ms]

    Conv2DSparse::ParamValues convParams(
        3, 3,           // conv_kh, conv_kw
        1, 1,           // conv_sh, conv_sw
        0, 0,           // conv_padh, conv_padw
        32, 32, 3,      // conv_ih, conv_iw, conv_ic
        30, 30, 32);    // conv_oh, conv_ow, conv_oc

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        initVar<InitVarSnippet::Kernel>());    // 0 - Wij (nA)

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::PoissonNew>("Poisson", 32 * 32 * 3,
                                                        poissonParams, poissonInit);
    model.addNeuronPopulation<NeuronModels::LIF>("Neurons", 30 * 30 * 32,
                                                 lifParams, lifInit);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn", SYNAPSE_MATRIX_TYPE, NO_DELAY,
        "Poisson", "Neurons",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<Conv2DSparse>(convParams));
}
