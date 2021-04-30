#pragma once

//----------------------------------------------------------------------------
// AvgPoolFixedNumberPreWithReplacement
//----------------------------------------------------------------------------
class AvgPoolFixedNumberPreWithReplacement : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(AvgPoolFixedNumberPreWithReplacement, 1);

    SET_COL_BUILD_CODE(
        "if(c == 0) {\n"
        "   $(endCol);\n"
        "}\n"
        "const unsigned int idPre = (unsigned int)ceil($(gennrand_uniform) * ($(num_pre) / 2) - 1;\n"
        "$(addSynapse, (idPre * 2) + $(id_pre_begin));\n"
        "$(addSynapse, (idPre * 2) + 1 + $(id_pre_begin));\n"
        "c--;\n");
    SET_COL_BUILD_STATE_VARS({{"c", "unsigned int", "$(colLength)"}});

    SET_PARAM_NAMES({"colLength"});

    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // In each column the number of connections that end up in a row are distributed
            // binomially with n=numConnections and p=1.0 / numPre. As there are numPost columns the total number
            // of connections that end up in each row are distributed binomially with n=numConnections * numPost and p=1.0 / numPre
            return binomialInverseCDF(quantile, (unsigned int)pars[0] * 2 * numPost, 1.0 / (double)numPre);
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::vector<double> &pars)
        {
            return 2 * (unsigned int)pars[0];
        });
};
IMPLEMENT_MODEL(AvgPoolFixedNumberPreWithReplacement)

//----------------------------------------------------------------------------
// InputNeuron
//----------------------------------------------------------------------------
class InputNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(InputNeuron, 2, 0);

    SET_SIM_CODE(
        "const int trial = (int)($(t) / $(presentMs));\n"
        "const uint8_t *imgData = &$(dataset)[trial * 44 * 16];\n"
        "const scalar u = $(gennrand_uniform);\n");
    SET_THRESHOLD_CONDITION_CODE("imgData[$(id)] > 0 && u >= exp(-(float)imgData[$(id)] * $(scale) * DT)");

    SET_PARAM_NAMES({"presentMs", "scale"});
    SET_EXTRA_GLOBAL_PARAMS({{"dataset", "uint8_t*"}});
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(InputNeuron);

//----------------------------------------------------------------------------
// WTAOutput
//----------------------------------------------------------------------------
class WTAOutput : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(WTAOutput, 1);

    SET_CODE(
        "$(value) = ($(id_pre) == $(id_post)) ? 0.0 : $(constant);\n");

    SET_PARAM_NAMES({"constant"});
};
IMPLEMENT_SNIPPET(WTAOutput);


//----------------------------------------------------------------------------
// Inf
//----------------------------------------------------------------------------
class Inf : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(Inf, 0, 0);

    SET_APPLY_INPUT_CODE(
        "$(Iinf) += $(inSyn);\n"
        "$(inSyn) = 0;\n");
};
IMPLEMENT_MODEL(Inf);
