#include "modelSpec.h"

//----------------------------------------------------------------------------
// WTA
//----------------------------------------------------------------------------
class WTA : public InitVarSnippet::Base
{
public:
    DECLARE_SNIPPET(WTA, 5);

    SET_CODE(
        "// Convert presynaptic neuron ID to rows, columns and channels\n"
        "const int inRow = ($(id_pre) / (int)$(conv_c)) / (int)$(conv_w);\n"
        "const int inCol = ($(id_pre) / (int)$(conv_c)) % (int)$(conv_w);\n"
        "const int inChan = $(id_pre) % (int)$(conv_c);\n"
        "// Convert postsynaptic neuron ID to rows, columns and channels\n"
        "const int outRow = ($(id_post) / (int)$(conv_c)) / (int)$(conv_w);\n"
        "const int outCol = ($(id_post) / (int)$(conv_c)) % (int)$(conv_w);\n"
        "const int outChan = $(id_post) % (int)$(conv_c);\n"
        "// If intra-map\n"
        "if(inChan == outChan) {\n"
        "   $(value) = (inRow == outRow && inCol == outCol) ? 0.0 : $(constant);\n"
        "}\n"
        "// Otherwise (inter-map)\n"
        "else {\n"
        "   const int rowDist = abs(inRow - outRow);\n"
        "   const int colDist = abs(inCol - outCol);\n"
        "   if(rowDist <= (int)$(radius) && colDist <= (int)$(radius)"
        "      && rowDist != 0 && colDist != 0)\n"
        "   {\n"
        "       $(value) = $(constant);\n"
        "   }\n"
        "   else {\n"
        "       $(value) = 0.0;\n"
        "   }\n"
        "}\n");

    SET_PARAM_NAMES({"conv_h", "conv_w", "conv_c", "radius", "constant"});
};
IMPLEMENT_SNIPPET(WTA);

//----------------------------------------------------------------------------
// InputNeuron
//----------------------------------------------------------------------------
class InputNeuron : public NeuronModels::Base
{
public:
    DECLARE_MODEL(InputNeuron, 0, 1);
    
    SET_VARS({{"input", "scalar", VarAccess::READ_ONLY_DUPLICATE}});
    
    SET_SIM_CODE("const scalar u = $(gennrand_uniform);\n");
    SET_THRESHOLD_CONDITION_CODE("$(input) > 0.0 && u >= exp(-$(input) * DT)");
    
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(InputNeuron);

//----------------------------------------------------------------------------
// DualAccumulator
//----------------------------------------------------------------------------
class DualAccumulator : public NeuronModels::Base
{
public:
    DECLARE_MODEL(DualAccumulator, 2, 2);

    SET_SIM_CODE(
        "// Reset inference accumulator if it's crossed threshold\n"
        "if($(Vinf) > $(VthreshInf)) {\n"
        "   $(Vinf) = 0.0;\n"
        "}\n"
        "// Apply inference input current to both accumulators\n"
        "$(Vwta) += $(Iinf);\n"
        "$(Vinf) += $(Iinf);\n"
        "// If there's WTA input, reset WTA accumulator\n"
        "if($(Isyn) < 0.0) {\n"
        "   $(Vwta) = 0.0;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(Vwta) > $(VthreshWTA)");

    SET_RESET_CODE("$(Vwta) = 0.0;\n");

    SET_PARAM_NAMES({
        "VthreshWTA",   // WTA threshold
        "VthreshInf"}); // Inference threshold
        
    SET_VARS({{"Vwta", "scalar"}, {"Vinf", "scalar"}});
    
    SET_ADDITIONAL_INPUT_VARS({{"Iinf", "scalar", "0.0"}});
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(DualAccumulator);

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

//----------------------------------------------------------------------------
// STDPBase
//----------------------------------------------------------------------------
class STDPBase : public WeightUpdateModels::Base
{
public:
    SET_PARAM_NAMES({
      "alphaPlus",  // 0 - Potentiation rate
      "alphaMinus", // 1 - Depression rate
      "betaPlus",   // 2 - Damping factor
      "Wmin",       // 3 - Minimum weight
      "Wmax"});     // 4 - Maximum weight

    SET_VARS({{"g", "scalar"}});
    
    
    SET_LEARN_POST_CODE(
        "if($(sT_pre) > $(prev_sT_post)) {\n"
        "   scalar old = $(g);\n"
        "   scalar assumed;\n"
        "   do {\n"
        "       assumed = old;\n"
        "       old = atomicCAS(&$(g), assumed, fmin($(Wmax), assumed + ($(alphaPlus) * exp(-$(betaPlus) * assumed))));\n"
        "   } while(assumed != old)\n"
        "}\n"
        "else {\n"
        "   scalar old = $(g);\n"
        "   scalar assumed;\n"
        "   do {\n"
        "       assumed = old;\n"
        "       old = atomicCAS(&$(g), assumed, fmax($(Wmin), assumed + $(alphaMinus)));\n"
        "   } while(assumed != old)\n"
        "}\n");

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_PREV_POST_SPIKE_TIME(true);
};

//----------------------------------------------------------------------------
// STDPInput
//----------------------------------------------------------------------------
class STDPInput : public STDPBase
{
public:
    DECLARE_MODEL(STDPInput, 5, 1);
    
    SET_SIM_CODE("$(addToInSyn, $(g));\n")
};
IMPLEMENT_MODEL(STDPInput);

//----------------------------------------------------------------------------
// STDP
//----------------------------------------------------------------------------
class STDP : public STDPBase
{
public:
    SET_EVENT_CODE("$(addToInSyn, $(g));\n");
    SET_EVENT_THRESHOLD_CONDITION_CODE("$(Vinf_pre) > $(VthreshInf_pre)");
};

void modelDefinition(ModelSpec &model)
{
    model.setDT(1.0);
    model.setName("deep_unsupervised_learning");
    model.setTiming(true);

    DualAccumulator::ParamValues convOneParams(
        8.0,    // VthreshWTA
        8.0);   // VthreshInf

    DualAccumulator::VarValues dualAccumulatorInitVals(
        0.0,    // Vwta
        0.0);   // Vinf

    InitSparseConnectivitySnippet::Conv2D::ParamValues conv1Params(
        5, 5,           // conv_kh, conv_kw
        1, 1,           // conv_sh, conv_sw
        0, 0,           // conv_padh, conv_padw
        28, 28, 1,      // conv_ih, conv_iw, conv_ic
        24, 24, 16);    // conv_oh, conv_ow, conv_oc
    
    WTA::ParamValues conv1WTAParams(
        5, 5, 16,   // conv_h, conv_w, conv_c
        2,          // radius
        1.0);       // constant
    
    STDPInput::ParamValues inputConv1Params(
      0.1,          // 0 - Potentiation rate
      0.1 / -8.0,   // 1 - Depression rate
      3.0,          // 2 - Damping factor
      0.0,          // 3 - Minimum weight
      1.0);         // 4 - Maximum weight

    model.addNeuronPopulation<InputNeuron>("Input", 28 * 28 * 1,
                                           {}, { uninitialisedVar() });
    model.addNeuronPopulation<DualAccumulator>("Conv1", 24 * 24 * 16,
                                               convOneParams, dualAccumulatorInitVals);

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Conv1_Conv1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Conv1", "Conv1",
        {}, initVar<WTA>(conv1WTAParams),
        {}, {});
    
    model.addSynapsePopulation<STDPInput, Inf>(
        "Input_Conv1", SynapseMatrixType::PROCEDURAL_KERNELG, NO_DELAY,
        "Input", "Conv1",
        inputConv1Params, { uninitialisedVar() },
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::Conv2D>(conv1Params));
}