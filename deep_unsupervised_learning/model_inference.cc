#include "modelSpec.h"

#include "models.h"
#include "parameters.h"

//----------------------------------------------------------------------------
// InferenceAccumulator
//----------------------------------------------------------------------------
class InferenceAccumulator : public NeuronModels::Base
{
public:
    DECLARE_MODEL(InferenceAccumulator, 1, 1);

    SET_SIM_CODE(
        "// Apply input current to inference accumulators\n"
        "$(Vinf) += $(Isyn);\n");

    SET_THRESHOLD_CONDITION_CODE("$(Vinf) > $(VthreshInf)");

    SET_RESET_CODE("$(Vinf) = 0.0;\n");

    SET_PARAM_NAMES({"VthreshInf"}); // Inference threshold

    SET_VARS({{"Vinf", "scalar"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(InferenceAccumulator);

//----------------------------------------------------------------------------
// WTAAccumulatorSpikeCount
//----------------------------------------------------------------------------
class WTAAccumulatorSpikeCount : public NeuronModels::Base
{
public:
    DECLARE_MODEL(WTAAccumulatorSpikeCount, 1, 2);

    SET_SIM_CODE(
        "if(fmod($(t), 100.0) < DT) {\n"
        "   $(SpikeCount) = 0;\n"
        "}\n"
        "// Apply input current to WTA accumulators\n"
        "$(Vwta) += $(Iinf);\n"
        "// If there's WTA input, reset WTA accumulator and time\n"
        "if($(Isyn) < 0.0) {\n"
        "   $(Vwta) = 0.0;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(Vwta) > $(VthreshWTA)");

    SET_RESET_CODE(
        "$(Vwta) = 0.0;\n"
        "$(SpikeCount)++;\n");

    SET_PARAM_NAMES({"VthreshWTA"}); // WTA threshold

    SET_VARS({{"Vwta", "scalar"}, {"SpikeCount", "unsigned int"}});
    SET_ADDITIONAL_INPUT_VARS({{"Iinf", "scalar", "0.0"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(WTAAccumulatorSpikeCount);

void modelDefinition(ModelSpec &model)
{
    using namespace Parameters;
    
    model.setDT(timestepMs);
    model.setName("deep_unsupervised_learning_inference");
    model.setTiming(true);
    
    InputNeuron::ParamValues inputParams(
        Input::presentMs,   // present time (ms)
        Input::scale);      // scale

    InferenceAccumulator::ParamValues conv1Params(
        Conv1::threshInf);  // VthreshInf
    
    InferenceAccumulator::ParamValues conv2Params(
        Conv2::threshInf);  // VthreshInf

    WTAAccumulatorSpikeCount::ParamValues outputParams(
        Output::threshWTA);  // VthreshWTA

    InferenceAccumulator::VarValues inferenceAccumulatorInitVals(
        0.0);   // Vinf
    
    WTAAccumulatorSpikeCount::VarValues outputInitVals(
        0.0,    // Vwta
        0);     // SpikeCount
    
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

    AvgPoolDense::ParamValues conv2OutputConnectParams(
        Conv2Output::poolKernelHeight, Conv2Output::poolKernelWidth,    // pool_kh, pool_kw
        Conv2Output::poolStrideHeight, Conv2Output::poolStrideWidth,    // pool_sh, pool_sw
        0, 0,                                                           // pool_padh, pool_padw
        Conv2::height, Conv2::width, Conv2::channels,                   // pool_ih, pool_iw, pool_ic
        Conv2Output::denseInHeight, Conv2Output::denseInWidth,          // dense_ih, dense_iw
        Output::numNeurons);                                            // dense_units
    
    WTAOutput::ParamValues outputWTAParams(
        -1.0);  // constant

    auto *input = model.addNeuronPopulation<InputNeuron>("Input", Input::numNeurons,
                                                         inputParams, {});
    auto *conv1 = model.addNeuronPopulation<InferenceAccumulator>("Conv1", Conv1::numNeurons,
                                                                  conv1Params, inferenceAccumulatorInitVals);
    auto *conv2 = model.addNeuronPopulation<InferenceAccumulator>("Conv2", Conv2::numNeurons,
                                                                  conv2Params, inferenceAccumulatorInitVals);
    auto *output = model.addNeuronPopulation<WTAAccumulatorSpikeCount>("Output", Output::numNeurons,
                                                                       outputParams, outputInitVals);
    input->setSpikeRecordingEnabled(true);
    conv1->setSpikeRecordingEnabled(true);
    conv2->setSpikeRecordingEnabled(true);
    output->setSpikeRecordingEnabled(true);
    
    // Add WTA connectivity
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Output_Output", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Output", "Output",
        {}, initVar<WTAOutput>(outputWTAParams),
        {}, {});
        
    // Add feedforward connecivity
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Input_Conv1", SynapseMatrixType::PROCEDURAL_KERNELG, NO_DELAY,
        "Input", "Conv1",
        {}, { uninitialisedVar() },
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::Conv2D>(inputConv1ConnectParams));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "Conv1_Conv2", SynapseMatrixType::PROCEDURAL_KERNELG, NO_DELAY,
        "Conv1", "Conv2",
        {}, { uninitialisedVar() },
        {}, {},
        initConnectivity<InitSparseConnectivitySnippet::AvgPoolConv2D>(conv1Conv2ConnectParams));

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, Inf>(
        "Conv2_Output", SynapseMatrixType::PROCEDURAL_KERNELG, NO_DELAY,
        "Conv2", "Output",
        {}, { uninitialisedVar() },
        {}, {},
        initConnectivity<AvgPoolDense>(conv2OutputConnectParams));
}