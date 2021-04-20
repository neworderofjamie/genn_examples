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
        "   if(rowDist <= (int)$(radius) && colDist <= (int)$(radius)) {"
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

//----------------------------------------------------------------------------
// DualAccumulator
//----------------------------------------------------------------------------
class DualAccumulator : public NeuronModels::Base
{
public:
    DECLARE_MODEL(DualAccumulator, 2, 3);

    SET_SIM_CODE(
        "// Reset inference accumulator if it's crossed threshold\n"
        "if($(Vinf) > $(VthreshInf)) {\n"
        "   $(Vinf) = 0.0;\n"
        "}\n"
        "// Apply inference input current to both accumulators\n"
        "$(Vwta) += $(Iinf);\n"
        "$(Vinf) += $(Iinf);\n"
        "// If there's WTA input, reset WTA accumulator and time\n"
        "if($(Isyn) < 0.0) {\n"
        "   $(Vwta) = 0.0;\n"
        "   $(TlastReset) = $(t);\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(Vwta) > $(VthreshWTA)");

    SET_RESET_CODE("$(Vwta) = 0.0;\n");

    SET_PARAM_NAMES({
        "VthreshWTA",   // WTA threshold
        "VthreshInf"}); // Inference threshold
        
    SET_VARS({{"Vwta", "scalar"}, {"Vinf", "scalar"}, {"TlastReset", "scalar"}});
    
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
// STDPOutput
//----------------------------------------------------------------------------
class STDPOutput : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(STDPOutput, 5, 1);
    
    SET_PARAM_NAMES({
      "alphaPlus",  // 0 - Potentiation rate
      "alphaMinus", // 1 - Depression rate
      "betaPlus",   // 2 - Damping factor
      "Wmin",       // 3 - Minimum weight
      "Wmax"});     // 4 - Maximum weight
  
    SET_VARS({{"g", "scalar"}});
  
    // **TODO** output layer will eventually pass spike-like-events instead
    SET_SIM_CODE("$(addToInSyn, $(g));\n");
  
    SET_LEARN_POST_CODE(
        "const scalar tPostLast = fmax($(prev_sT_post), $(TlastReset_post));\n"
        "if($(sT_pre) > tPostLast) {\n"
        "   $(g) = fmin($(Wmax), $(g) + ($(alphaPlus) * exp(-$(betaPlus) * $(g))));\n"
        "}\n"
        "else {\n"
        "   $(g) = fmax($(Wmin), $(g) + $(alphaMinus));\n"
        "}\n");
    
    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_PREV_POST_SPIKE_TIME(true);
};
IMPLEMENT_MODEL(STDPOutput);

//----------------------------------------------------------------------------
// STDPInput
//----------------------------------------------------------------------------
class STDPInput : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(STDPInput, 5, 1);
    
    SET_PARAM_NAMES({
      "alphaPlus",  // 0 - Potentiation rate
      "alphaMinus", // 1 - Depression rate
      "betaPlus",   // 2 - Damping factor
      "Wmin",       // 3 - Minimum weight
      "Wmax"});     // 4 - Maximum weight

    SET_VARS({{"g", "scalar"}});
    
    SET_SIM_CODE("$(addToInSyn, $(g));\n");
    
    SET_LEARN_POST_CODE(
        "const scalar tPostLast = fmax($(prev_sT_post), $(TlastReset_post));\n"
        "int *intAddr = (int*)&$(g);\n"
        "int old = *intAddr;\n"
        "int assumed;\n"
        "if($(sT_pre) > tPostLast) {\n"
        "   do {\n"
        "       assumed = old;\n"
        "       old = atomicCAS(intAddr, assumed, __float_as_int(fmin($(Wmax), __int_as_float(assumed) + ($(alphaPlus) * exp(-$(betaPlus) * __int_as_float(assumed))))));\n"
        "   } while(assumed != old);\n"
        "}\n"
        "else {\n"
        "   do {\n"
        "       assumed = old;\n"
        "       old = atomicCAS(intAddr, assumed, __float_as_int(fmax($(Wmin), __int_as_float(assumed) + $(alphaMinus))));\n"
        "   } while(assumed != old);\n"
        "}\n");

    SET_NEEDS_PRE_SPIKE_TIME(true);
    SET_NEEDS_PREV_POST_SPIKE_TIME(true);
};
IMPLEMENT_MODEL(STDPInput);

//----------------------------------------------------------------------------
// STDPHidden
//----------------------------------------------------------------------------
class STDPHidden : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(STDPHidden, 5, 1);
    
    SET_PARAM_NAMES({
      "alphaPlus",  // 0 - Potentiation rate
      "alphaMinus", // 1 - Depression rate
      "betaPlus",   // 2 - Damping factor
      "Wmin",       // 3 - Minimum weight
      "Wmax"});     // 4 - Maximum weight

    SET_VARS({{"g", "scalar"}});

    SET_EVENT_CODE("$(addToInSyn, $(g));\n");
    SET_EVENT_THRESHOLD_CONDITION_CODE("$(Vinf_pre) > $(VthreshInf_pre)");

    SET_LEARN_POST_CODE(
        "const scalar tPostLast = fmax($(prev_sT_post), $(TlastReset_post));\n"
        "int *intAddr = (int*)&$(g);\n"
        "int old = *intAddr;\n"
        "int assumed;\n"
        "if($(seT_pre) > tPostLast) {\n"
        "   do {\n"
        "       assumed = old;\n"
        "       old = atomicCAS(intAddr, assumed, __float_as_int(fmin($(Wmax), __int_as_float(assumed) + ($(alphaPlus) * exp(-$(betaPlus) * __int_as_float(assumed))))));\n"
        "   } while(assumed != old);\n"
        "}\n"
        "else {\n"
        "   do {\n"
        "       assumed = old;\n"
        "       old = atomicCAS(intAddr, assumed, __float_as_int(fmax($(Wmin), __int_as_float(assumed) + $(alphaMinus))));\n"
        "   } while(assumed != old);\n"
        "}\n");

    SET_NEEDS_PRE_SPIKE_EVENT_TIME(true);
    SET_NEEDS_PREV_POST_SPIKE_TIME(true);
};
IMPLEMENT_MODEL(STDPHidden);

void modelDefinition(ModelSpec &model)
{
    model.setDT(0.1);
    model.setName("deep_unsupervised_learning");
    model.setTiming(true);
    
    InputNeuron::ParamValues inputParams(
        100.0,  // presentMs
        0.00125); // scale

    DualAccumulator::ParamValues conv1Params(
        8.0,    // VthreshWTA
        8.0);   // VthreshInf
    
    DualAccumulator::ParamValues conv2Params(
        30.0,   // VthreshWTA
        30.0);  // VthreshInf
        
    DualAccumulator::ParamValues outputParams(
        30.0,    // VthreshWTA
        30.0);   // VthreshInf

    DualAccumulator::VarValues dualAccumulatorInitVals(
        0.0,                                    // Vwta
        0.0,                                    // Vinf
        -std::numeric_limits<float>::max());    // TlastReset

    InitSparseConnectivitySnippet::Conv2D::ParamValues inputConv1ConnectParams(
        5, 5,           // conv_kh, conv_kw
        1, 1,           // conv_sh, conv_sw
        0, 0,           // conv_padh, conv_padw
        28, 28, 1,      // conv_ih, conv_iw, conv_ic
        24, 24, 16);    // conv_oh, conv_ow, conv_oc
    
    InitSparseConnectivitySnippet::AvgPoolConv2D::ParamValues conv1Conv2ConnectParams(
        2, 2,       // pool_kh, pool_kw
        2, 2,       // pool_sh, pool_sw
        0, 0,       // pool_padh, pool_padw
        24, 24, 16, // pool_ih, pool_iw, pool_ic,
        5, 5,       // conv_kh, conv_kw
        1, 1,       // conv_sh, conv_sw
        0, 0,       // conv_padh, conv_padw
        12, 12,     // conv_ih, conv_iw
        8, 8, 32);  // conv_oh, conv_ow, conv_oc

    WTA::ParamValues conv1WTAParams(
        24, 24, 16, // conv_h, conv_w, conv_c
        2,          // radius
        -1.0);      // constant
    
     WTA::ParamValues conv2WTAParams(
        8, 8, 32,   // conv_h, conv_w, conv_c
        2,          // radius
        -1.0);      // constant

    WTAOutput::ParamValues outputWTAParams(
        -1.0);  // constant
    
    STDPInput::ParamValues inputConv1Params(
        0.0001,           // 0 - Potentiation rate
        0.0001 / -8.0,    // 1 - Depression rate
        3.0,            // 2 - Damping factor
        0.0,            // 3 - Minimum weight
        1.0);           // 4 - Maximum weight
    
    STDPHidden::ParamValues conv1Conv2Params(
        0.0001,         // 0 - Potentiation rate
        0.0001 / -8.0,  // 1 - Depression rate
        3.0,            // 2 - Damping factor
        0.0,            // 3 - Minimum weight
        1.0);           // 4 - Maximum weight

    /*STDPOutput::ParamValues inputOutputParams(
        0.0001,           // 0 - Potentiation rate
        0.0001 / -8.0,    // 1 - Depression rate
        3.0,            // 2 - Damping factor
        0.0,            // 3 - Minimum weight
        1.0);           // 4 - Maximum weight*/
    
    STDPOutput::VarValues inputOutputVals(
        initVar<InitVarSnippet::Normal>({0.67, 0.1}));  // g

    auto *input = model.addNeuronPopulation<InputNeuron>("Input", 28 * 28 * 1,
                                                         inputParams, {});
    auto *conv1 = model.addNeuronPopulation<DualAccumulator>("Conv1", 24 * 24 * 16,
                                                             conv1Params, dualAccumulatorInitVals);
    auto *conv2 = model.addNeuronPopulation<DualAccumulator>("Conv2", 8 * 8 * 32,
                                                             conv2Params, dualAccumulatorInitVals);
                         
    //auto *output = model.addNeuronPopulation<DualAccumulator>("Output", 1000,
    //                                                          outputParams, dualAccumulatorInitVals);
    input->setSpikeRecordingEnabled(true);
    conv1->setSpikeRecordingEnabled(true);
    conv2->setSpikeRecordingEnabled(true);
    //output->setSpikeRecordingEnabled(true);
    
    // Add WTA connectivity
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Conv1_Conv1", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Conv1", "Conv1",
        {}, initVar<WTA>(conv1WTAParams),
        {}, {});
    
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Conv2_Conv2", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Conv2", "Conv2",
        {}, initVar<WTA>(conv2WTAParams),
        {}, {});
    
    // Add plastic, feedforward connecivity
    model.addSynapsePopulation<STDPInput, Inf>(
        "Input_Conv1", SynapseMatrixType::PROCEDURAL_KERNELG, NO_DELAY,
        "Input", "Conv1",
        inputConv1Params, { uninitialisedVar() },
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::Conv2D>(inputConv1ConnectParams));
    
    model.addSynapsePopulation<STDPHidden, Inf>(
        "Conv1_Conv2", SynapseMatrixType::PROCEDURAL_KERNELG, NO_DELAY,
        "Conv1", "Conv2",
        conv1Conv2Params, { uninitialisedVar() },
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::AvgPoolConv2D>(conv1Conv2ConnectParams));
    
    /*model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Output_Output", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Output", "Output",
        {}, initVar<WTAOutput>(outputWTAParams),
        {}, {});
    
    model.addSynapsePopulation<STDPOutput, Inf>(
        "Input_Output", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Input", "Output",
        inputOutputParams, inputOutputVals,
        {}, {});*/
}