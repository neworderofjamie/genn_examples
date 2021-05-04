#pragma once

//----------------------------------------------------------------------------
// AvgPoolFixedNumberPreWithReplacement
//----------------------------------------------------------------------------
class AvgPoolFixedNumberPreWithReplacement : public InitSparseConnectivitySnippet::Base
{
public:
    DECLARE_SNIPPET(AvgPoolFixedNumberPreWithReplacement, 6);

    SET_COL_BUILD_CODE(
        "if(c == 0) {\n"
        "   $(endCol);\n"
        "}\n"
        "// Pick which pool region to connect\n"
        "const int poolInIdx = (unsigned int)ceil($(gennrand_uniform) * $(numPools) - 1);\n"
        "// Convert to row, column and channel\n"
        "const int poolInRow = (poolInIdx / (int)$(pool_ic)) / ((int)$(pool_iw) / (int)$(pool_kw));\n"
        "const int poolInCol = (poolInIdx / (int)$(pool_ic)) % ((int)$(pool_iw) / (int)$(pool_kw));\n"
        "const int poolInChan = poolInIdx % (int)$(pool_ic);\n"
        "// Add synapses across all inputs to pool\n"
        "$(addSynapse, ((poolInRow * 2) * (int)$(pool_iw) * (int)$(pool_ic)) + ((poolInCol * 2) * (int)$(pool_ic)) + poolInChan);\n"
        "$(addSynapse, (((poolInRow * 2) + 1) * (int)$(pool_iw) * (int)$(pool_ic)) + ((poolInCol * 2) * (int)$(pool_ic)) + poolInChan);\n"
        "$(addSynapse, (((poolInRow * 2) + 1) * (int)$(pool_iw) * (int)$(pool_ic)) + (((poolInCol * 2) + 1) * (int)$(pool_ic)) + poolInChan);\n"
        "$(addSynapse, ((poolInRow * 2) * (int)$(pool_iw) * (int)$(pool_ic)) + (((poolInCol * 2) + 1) * (int)$(pool_ic)) + poolInChan);\n"
        "c--;\n");
    SET_COL_BUILD_STATE_VARS({{"c", "unsigned int", "$(colLength)"}});

    SET_PARAM_NAMES({"colLength", 
                     "pool_kh", "pool_kw",
                     "pool_ih", "pool_iw", "pool_ic"});
    SET_DERIVED_PARAMS({{"numPools", [](const std::vector<double> &pars, double){ return ((int)pars[3] / (int)pars[1]) * ((int)pars[4] / (int)pars[2]) * (int)pars[5]; }}});
    
    SET_CALC_MAX_ROW_LENGTH_FUNC(
        [](unsigned int numPre, unsigned int numPost, const std::vector<double> &pars)
        {
            // Calculate suitable quantile for 0.9999 change when drawing numPre times
            const double quantile = pow(0.9999, 1.0 / (double)numPre);

            // In each column the number of connections that end up in a row are distributed
            // binomially with n=numConnections and p=1.0 / numPre. As there are numPost columns the total number
            // of connections that end up in each row are distributed binomially with n=numConnections * numPost and p=1.0 / numPre
            return binomialInverseCDF(quantile, (unsigned int)pars[0] * 4 * numPost, 1.0 / (double)numPre);
        });

    SET_CALC_MAX_COL_LENGTH_FUNC(
        [](unsigned int, unsigned int, const std::vector<double> &pars)
        {
            return 4 * (unsigned int)pars[0];
        });
};
IMPLEMENT_MODEL(AvgPoolFixedNumberPreWithReplacement);

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
