#pragma once

//----------------------------------------------------------------------------
// AvgPoolDense
//----------------------------------------------------------------------------
class AvgPoolDense : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(AvgPoolDense, 12);

    SET_PARAM_NAMES({"pool_kh", "pool_kw",
                     "pool_sh", "pool_sw",
                     "pool_padh", "pool_padw",
                     "pool_ih", "pool_iw", "pool_ic",
                     "dense_ih", "dense_iw",
                     "dense_units"});

     SET_ROW_BUILD_STATE_VARS({{"poolInRow", "int", "($(id_pre) / (int)$(pool_ic)) / (int)$(pool_iw)"},
                               {"poolInCol", "int", "($(id_pre) / (int)$(pool_ic)) % (int)$(pool_iw)"},
                               {"poolInChan", "int", "$(id_pre) % (int)$(pool_ic)"},
                               {"poolOutRow", "int", "(poolInRow + (int)$(pool_padh)) / (int)$(pool_sh)"},
                               {"poolStrideRow", "int", "(poolOutRow * (int)$(pool_sh)) - (int)$(pool_padh)"},
                               {"poolOutCol", "int", "(poolInCol + (int)$(pool_padw)) / (int)$(pool_sw)"},
                               {"poolStrideCol", "int", "(poolOutCol * (int)$(pool_sw)) - (int)$(pool_padw)"},
                               {"denseInIdx", "int", "(poolOutRow * (int)$(dense_iw) * (int)$(pool_ic)) + (poolOutCol * (int)$(pool_ic)) + poolInChan"},
                               {"synIdx", "int", "(int)$(dense_units) * denseInIdx"},
                               {"denseOutIdx", "int", 0}});

    SET_COL_BUILD_STATE_VARS({{"numDenseIn", "int", "(int)$(dense_iw) * (int)$(dense_ih) * (int)$(pool_ic)"},
                              {"denseInIdx", "int", 0},
                              {"synIdx", "int", "$(id_post)"}});

    SET_ROW_BUILD_CODE(
        "if(($(poolInRow) >= ($(poolStrideRow) + (int)$(pool_kh))) || ($(poolInCol) >= ($(poolStrideCol) + (int)$(pool_kw)))) {\n"
        "   $(endRow);\n"
        "}\n"
        "if($(denseOutIdx) == (int)$(dense_units)) {\n"
        "   $(endRow);\n"
        "}\n"
        "$(addSynapse, $(denseOutIdx), $(synIdx));\n"
        "$(synIdx)++;\n"
        "$(denseOutIdx)++;\n");

    SET_COL_BUILD_CODE(
        "if($(denseInIdx) == $(numDenseIn)) {\n"
        "   $(endCol);\n"
        "}\n"
        "const int denseInRow = (denseInIdx / (int)$(pool_ic)) / (int)$(dense_iw);\n"
        "const int denseInCol = (denseInIdx / (int)$(pool_ic)) % (int)$(dense_iw);\n"
        "const int poolInChan = denseInIdx % (int)$(pool_ic);\n"
        "const int poolInRow = (denseInRow * (int)$(pool_sh)) - (int)$(pool_padh);\n"
        "const int poolInCol = (denseInCol * (int)$(pool_sw)) - (int)$(pool_padw);\n"
        "const int idPre = ((poolInRow * (int)$(pool_iw) * (int)$(pool_ic)) +\n"
        "                   (poolInCol * (int)$(pool_ic)) +\n"
        "                   poolInChan);\n"
        "$(addSynapse, idPre, $(synIdx));\n"
        "$(denseInIdx)++;\n"
        "$(synIdx) += (int)$(dense_units);\n");

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::vector<double> &pars)
        {
            return (unsigned int)pars[11];
        });

    SET_CALC_KERNEL_SIZE_FUNC(
        [](const std::vector<double> &pars)->std::vector<unsigned int>
        {
            const unsigned int poolIC = (unsigned int)pars[8];
            const unsigned int denseIH = (unsigned int)pars[9];
            const unsigned int denseIW = (unsigned int)pars[10];
            const unsigned int denseUnits = (unsigned int)pars[11];
            return {poolIC * denseIH * denseIW * denseUnits};
        });
};
IMPLEMENT_SNIPPET(AvgPoolDense);

//----------------------------------------------------------------------------
// InputNeuron
//----------------------------------------------------------------------------
class InputNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(InputNeuron, 2, 0);

    SET_SIM_CODE(
        "const int trial = (int)($(t) / $(presentMs));\n"
        "const uint8_t *imgData = &$(dataset)[trial * 28 * 28];\n"
        "const scalar u = $(gennrand_uniform);\n");
    SET_THRESHOLD_CONDITION_CODE("imgData[$(id)] > 0 && u >= exp(-(float)imgData[$(id)] * $(scale) * DT)");

    SET_PARAM_NAMES({"presentMs", "scale"});
    SET_EXTRA_GLOBAL_PARAMS({{"dataset", "uint8_t*"}});
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(InputNeuron);