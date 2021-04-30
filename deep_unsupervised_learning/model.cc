#include "modelSpec.h"

#include "models.h"
#include "parameters.h"

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
// STDPInput
//----------------------------------------------------------------------------
//! STDP model for connections from input layer - passes spikes
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
//! STDP model for connections from hidden layers - passes spike-like events
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
    using namespace Parameters;
    
    model.setDT(timestepMs);
    model.setName("deep_unsupervised_learning");
    model.setTiming(true);

    InputNeuron::ParamValues inputParams(
        Input::presentMs,   // present time (ms)
        Input::scale);      // scale

    DualAccumulator::ParamValues conv1Params(
        Conv1::threshWTA,   // VthreshWTA
        Conv1::threshInf);  // VthreshInf
    
    DualAccumulator::ParamValues conv2Params(
        Conv2::threshWTA,   // VthreshWTA
        Conv2::threshInf);  // VthreshInf

    //DualAccumulator::ParamValues outputParams(
    //    Output::threshWTA,   // VthreshWTA
    //    Output::threshInf);  // VthreshInf

    DualAccumulator::VarValues dualAccumulatorInitVals(
        0.0,                                    // Vwta
        0.0,                                    // Vinf
        -std::numeric_limits<float>::max());    // TlastReset
    
    InitSparseConnectivitySnippet::Conv2D::ParamValues inputConv1ConnectParams(
        InputConv1::convKernelHeight, InputConv1::convKernelWidth,  // conv_kh, conv_kw
        InputConv1::convStrideHeight, InputConv1::convStrideWidth,  // conv_sh, conv_sw
        0, 0,                                                       // conv_padh, conv_padw
        Input::height, Input::width, Input::channels,               // conv_ih, conv_iw, conv_ic
        Conv1::height, Conv1::width, InputConv1::numFilters);       // conv_oh, conv_ow, conv_oc

    InitSparseConnectivitySnippet::AvgPoolConv2D::ParamValues conv1Conv2ConnectParams(
        Conv1Conv2::poolKernelHeight, Conv1Conv2::poolKernelWidth,  // pool_kh, pool_kw
        Conv1Conv2::poolStrideHeight, Conv1Conv2::poolStrideWidth,  // pool_sh, pool_sw
        0, 0,                                                       // pool_padh, pool_padw
        Conv1::height, Conv1::width, Conv1::channels,               // pool_ih, pool_iw, pool_ic,
        Conv1Conv2::convKernelHeight, Conv1Conv2::convKernelWidth,  // conv_kh, conv_kw
        Conv1Conv2::convStrideHeight, Conv1Conv2::convStrideWidth,  // conv_sh, conv_sw
        0, 0,                                                       // conv_padh, conv_padw
        Conv1Conv2::convInHeight, Conv1Conv2::convInWidth,          // conv_ih, conv_iw
        Conv2::height, Conv2::width, Conv1Conv2::numFilters);       // conv_oh, conv_ow, conv_oc

    /*AvgPoolDense::ParamValues conv2OutputConnectParams(
        Conv2Output::poolKernelHeight, Conv2Output::poolKernelWidth,    // pool_kh, pool_kw
        Conv2Output::poolStrideHeight, Conv2Output::poolStrideWidth,    // pool_sh, pool_sw
        0, 0,                                                           // pool_padh, pool_padw
        Conv2::height, Conv2::width, Conv2::channels,                   // pool_ih, pool_iw, pool_ic
        Conv2Output::denseInHeight, Conv2Output::denseInWidth,          // dense_ih, dense_iw
        Output::numNeurons);                                            // dense_units*/

    WTA::ParamValues conv1WTAParams(
        Conv1::height, Conv1::width, Conv1::channels, // conv_h, conv_w, conv_c
        Conv1::WTARadius,                             // radius
        -1.0);      // constant

     WTA::ParamValues conv2WTAParams(
        Conv2::height, Conv2::width, Conv2::channels,   // conv_h, conv_w, conv_c
        Conv2::WTARadius,                               // radius
        -1.0);      // constant

    //WTAOutput::ParamValues outputWTAParams(
    //    -1.0);  // constant

    STDPInput::ParamValues inputConv1Params(
        0.001,          // 0 - Potentiation rate
        0.001 / -8.0,   // 1 - Depression rate
        3.0,            // 2 - Damping factor
        0.0,            // 3 - Minimum weight
        1.0);           // 4 - Maximum weight

    STDPHidden::ParamValues conv1Conv2Params(
        0.001,          // 0 - Potentiation rate
        0.001 / -8.0,   // 1 - Depression rate
        3.0,            // 2 - Damping factor
        0.0,            // 3 - Minimum weight
        1.0);           // 4 - Maximum weight

    STDPHidden::ParamValues conv2OutputParams(
        0.001,          // 0 - Potentiation rate
        0.001 / -8.0,   // 1 - Depression rate
        3.0,            // 2 - Damping factor
        0.0,            // 3 - Minimum weight
        1.0);           // 4 - Maximum weight

    auto *input = model.addNeuronPopulation<InputNeuron>("Input", Input::numNeurons,
                                                         inputParams, {});
    auto *conv1 = model.addNeuronPopulation<DualAccumulator>("Conv1", Conv1::numNeurons,
                                                             conv1Params, dualAccumulatorInitVals);
    auto *conv2 = model.addNeuronPopulation<DualAccumulator>("Conv2", Conv2::numNeurons,
                                                             conv2Params, dualAccumulatorInitVals);
    //auto *output = model.addNeuronPopulation<DualAccumulator>("Output", Output::numNeurons,
    //                                                          outputParams, dualAccumulatorInitVals);
    input->setSpikeRecordingEnabled(true);
    conv1->setSpikeRecordingEnabled(true);
    conv1->setSpikeEventRecordingEnabled(true);
    conv2->setSpikeRecordingEnabled(true);
    //conv2->setSpikeEventRecordingEnabled(true);
    //output->setSpikeRecordingEnabled(true);
    //output->setSpikeEventRecordingEnabled(true);

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

    //model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
    //    "Output_Output", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
    //    "Output", "Output",
    //    {}, initVar<WTAOutput>(outputWTAParams),
    //    {}, {});

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

    /*model.addSynapsePopulation<STDPHidden, Inf>(
        "Conv2_Output", SynapseMatrixType::PROCEDURAL_KERNELG, NO_DELAY,
        "Conv2", "Output",
        conv2OutputParams, { uninitialisedVar() },
        {}, {},
        initConnectivity<AvgPoolDense>(conv2OutputConnectParams));*/
}
