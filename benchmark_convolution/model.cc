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

    SET_ROW_BUILD_STATE_VARS({{"inRow", "unsigned int", "($(id_pre) / (unsigned int)$(conv_ic)) / (unsigned int)$(conv_iw)"},
                              {"inCol", "unsigned int", "($(id_pre) / (unsigned int)$(conv_ic)) % (unsigned int)$(conv_iw)"},
                              {"inChan", "unsigned int", "$(id_pre) % (unsigned int)$(conv_ic)"},
                              {"outRow", "unsigned int", "min((unsigned int)$(conv_oh), max(0, 1 + ((inRow + (unsigned int)$(conv_padh) - (unsigned int)$(conv_kh)) / (unsigned int)$(conv_sh))))"},
                              {"maxOutRow", "unsigned int", "min((unsigned int)$(conv_oh), max(0, 1 + ((inRow + (unsigned int)$(conv_padh)) / (unsigned int)$(conv_sh))))"},
                              {"minOutCol", "unsigned int", "min((unsigned int)$(conv_ow), max(0, 1 + ((inCol + (unsigned int)$(conv_padw) - (unsigned int)$(conv_kw)) / (unsigned int)$(conv_sw))))"},
                              {"maxOutCol", "unsigned int", "min((unsigned int)$(conv_ow), max(0, 1 + ((inCol + (unsigned int)$(conv_padw)) / (unsigned int)$(conv_sw))))"}});

    SET_ROW_BUILD_CODE(
        "if($(outRow) == $(maxOutRow)) {\n"
        "   $(endRow);\n"
        "}\n"
        "const unsigned int strideRow = ($(outRow) * $(conv_sh)) - $(conv_padh);\n"
        "const unsigned int kernRow = $(inRow) - strideRow;\n"
        "for(unsigned int outCol = $(minOutCol); outCol < $(maxOutCol); outCol++) {\n"
        "    const unsigned int strideCol = (outCol * $(conv_sw)) - $(conv_padw);\n"
        "    const unsigned int kernCol = $(inCol) - strideCol;\n"
        "    for(unsigned int outChan = 0; outChan < (unsigned int)$(conv_oc); outChan++) {\n"
        "        const unsigned int idPost = (($(outRow) * $(conv_ow) * $(conv_oc)) +\n"
        "                                     (outCol * $(conv_oc)) +\n"
        "                                     outChan);\n"
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

/*class Conv2D : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(Conv2D, 12);

    SET_PARAM_NAMES({"conv_kh", "conv_kw",
                     "conv_sh", "conv_sw",
                     "conv_padh", "conv_padw",
                     "conv_ih", "conv_iw", "conv_ic",
                     "conv_oh", "conv_ow", "conv_oc"});
    SET_GROUP_PARAMS({{"conv_kh_reg", "int", "$(conv_kh)"},
                      {"conv_kw_reg", "int", "$(conv_kw)"},
                      {"conv_sh_reg", "int", "$(conv_sh)"},
                      {"conv_sw_reg", "int", "$(conv_sw)"},
                      {"conv_padh_reg", "int", "$(conv_padh)"},
                      {"conv_padw_reg", "int", "$(conv_padw)"},
                      {"conv_iw_reg", "int", "$(conv_iw)"},
                      {"conv_ic_reg", "int", "$(conv_ic)"},
                      {"conv_ow_reg", "int", "$(conv_ow)"},
                      {"conv_oc_reg", "int", "$(conv_oc)"}});
    SET_PRE_PARAMS({{"conv_in_row", "int", "($(id_pre) / $(conv_ic_reg)) / $(conv_iw_reg)"},
                    {"conv_in_col", "int", "($(id_pre) / $(conv_ic_reg)) % $(conv_iw_reg)"},
                    {"conv_in_chan", "int", "$(id_pre) % $(conv_ic_reg)"}});
    SET_POST_PARAMS({{"conv_out_row", "int", "($(id_post) / $(conv_oc_reg)) / $(conv_ow_reg)"},
                     {"conv_out_col", "int", "($(id_post) / $(conv_oc_reg)) % $(conv_ow_reg)"},
                     {"conv_out_chan", "int", "$(id_post) % $(conv_oc_reg)"}});

    SET_EXTRA_GLOBAL_PARAMS({{"kernels", "scalar*"}});

    SET_CODE(
        "const int conv_stride_row = $(conv_out_row) * $(conv_sh_reg) - $(conv_padh_reg);\n"
        "const int conv_stride_col = $(conv_out_col) * $(conv_sw_reg) - $(conv_padw_reg);\n"
        "const int conv_k_row = $(conv_in_row) - conv_stride_row;\n"
        "const int conv_k_col = $(conv_in_col) - conv_stride_col;\n"
        "if (conv_k_row >= 0 && conv_k_row < $(conv_kh_reg) && conv_k_col >= 0 && conv_k_col < $(conv_kw_reg)) {\n"
        "    $(value) = $(kernels)[\n"
        "        conv_k_row * ($(conv_kw_reg) * $(conv_ic_reg) * $(conv_oc_reg)) +\n"
        "        conv_k_col * ($(conv_ic_reg) * $(conv_oc_reg)) +\n"
        "        $(conv_in_chan) * $(conv_oc_reg) +\n"
        "        $(conv_out_chan)\n"
        "    ];\n"
        "}\n"
        "else {\n"
        "    $(value) = 0.0;\n"
        "}");
};
IMPLEMENT_SNIPPET(Conv2D);*/

class Conv2D : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(Conv2D, 12);

    SET_PARAM_NAMES({"conv_kh", "conv_kw",
                     "conv_sh", "conv_sw",
                     "conv_padh", "conv_padw",
                     "conv_ih", "conv_iw", "conv_ic",
                     "conv_oh", "conv_ow", "conv_oc"});

    SET_EXTRA_GLOBAL_PARAMS({{"kernels", "scalar*"}});

    SET_CODE(
        "const int conv_kh = $(conv_kh), conv_kw = $(conv_kw);\n"
        "const int conv_sh = $(conv_sh), conv_sw = $(conv_sw);\n"
        "const int conv_padh = $(conv_padh), conv_padw = $(conv_padw);\n"
        "const int conv_iw = $(conv_iw), conv_ic = $(conv_ic);\n"
        "const int conv_ow = $(conv_ow), conv_oc = $(conv_oc);\n"
        "const int conv_in_row = ($(id_pre) / conv_ic) / conv_iw;\n"
        "const int conv_in_col = ($(id_pre) / conv_ic) % conv_iw;\n"
        "const int conv_in_chan = $(id_pre) % conv_ic;\n"
        "const int conv_out_row = ($(id_post) / conv_oc) / conv_ow;\n"
        "const int conv_out_col = ($(id_post) / conv_oc) % conv_ow;\n"
        "const int conv_out_chan = $(id_post) % conv_oc;\n"
        "int conv_stride_row = conv_out_row * conv_sh - conv_padh;\n"
        "int conv_stride_col = conv_out_col * conv_sw - conv_padw;\n"
        "int conv_k_row = conv_in_row - conv_stride_row;\n"
        "int conv_k_col = conv_in_col - conv_stride_col;\n"
        "if (conv_k_row >= 0 && conv_k_row < conv_kh && conv_k_col >= 0 && conv_k_col < conv_kw) {\n"
        "    $(value) = $(kernels)[\n"
        "        conv_k_row * (conv_kw * conv_ic * conv_oc) +\n"
        "        conv_k_col * (conv_ic * conv_oc) +\n"
        "        conv_in_chan * (conv_oc) +\n"
        "        conv_out_chan\n"
        "    ];\n"
        "}\n"
        "else {\n"
        "    $(value) = 0.0;\n"
        "}");
};
IMPLEMENT_SNIPPET(Conv2D);

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

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_PROCEDURAL
    Conv2DSparse::ParamValues convParams(
        3, 3,           // conv_kh, conv_kw
        1, 1,           // conv_sh, conv_sw
        0, 0,           // conv_padh, conv_padw
        32, 32, 3,      // conv_ih, conv_iw, conv_ic
        30, 30, 32);    // conv_oh, conv_ow, conv_oc

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        uninitialisedVar());    // 0 - Wij (nA)
#else
    Conv2D::ParamValues convParams(
        3, 3,           // conv_kh, conv_kw
        1, 1,           // conv_sh, conv_sw
        0, 0,           // conv_padh, conv_padw
        32, 32, 3,      // conv_ih, conv_iw, conv_ic
        30, 30, 32);    // conv_oh, conv_ow, conv_oc*/

    // Static synapse parameters
    WeightUpdateModels::StaticPulse::VarValues staticSynapseInit(
        initVar<Conv2D>(convParams));    // 0 - Wij (nA)
#endif

    // Create IF_curr neuron
    model.addNeuronPopulation<NeuronModels::PoissonNew>("Poisson", 32 * 32 * 3,
                                                        poissonParams, poissonInit);
    model.addNeuronPopulation<NeuronModels::LIF>("Neurons", 30 * 30 * 32,
                                                 lifParams, lifInit);

#ifdef SYNAPSE_MATRIX_CONNECTIVITY_PROCEDURAL
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn", SYNAPSE_MATRIX_TYPE, NO_DELAY,
        "Poisson", "Neurons",
        {}, staticSynapseInit,
        {}, {},
        initConnectivity<Conv2DSparse>(convParams));
#else
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Syn", SYNAPSE_MATRIX_TYPE, NO_DELAY,
        "Poisson", "Neurons",
        {}, staticSynapseInit,
        {}, {});
#endif
}
