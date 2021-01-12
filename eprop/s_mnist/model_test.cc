#include "modelSpec.h"

#include "../common/eprop_models.h"

#include "models.h"
#include "parameters.h"

//----------------------------------------------------------------------------
// OutputClassification
//----------------------------------------------------------------------------
class OutputClassification : public NeuronModels::Base
{
public:
    DECLARE_MODEL(OutputClassification, 1, 3);

    SET_PARAM_NAMES({"TauOut"});    // Membrane time constant [ms]

    SET_VARS({{"Y", "scalar"}, {"Pi", "scalar"}, {"B", "scalar", VarAccess::READ_ONLY}});

    SET_DERIVED_PARAMS({
        {"Kappa", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});

    SET_SIM_CODE(
        "$(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(B);\n"
        "scalar m = $(Y);\n"
        "m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x1));\n"
        "m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x2));\n"
        "m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x4));\n"
        "m = fmax(m, __shfl_xor_sync(0xFFFF, m, 0x8));\n"
        "const scalar expPi = exp($(Y) - m);\n"
        "scalar sumExpPi = expPi;\n"
        "sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x1);\n"
        "sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x2);\n"
        "sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x4);\n"
        "sumExpPi +=  __shfl_xor_sync(0xFFFF, sumExpPi, 0x8);\n"
        "$(Pi) = expPi / sumExpPi;\n");

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(OutputClassification);

void modelDefinition(ModelSpec &model)
{
    // Calculate weight scaling factor
    // **NOTE** "Long short-term memory and 
    // learning-to-learn in networks of spiking neurons"
    // suggests that this should be (1 Volt * DT)/Rm but
    // that results in 1E-9 or something which is never 
    // going to make these neurons spike - the paper then 
    // SORT OF suggests that they use 1.0
    const double weight0 = 1.0;

    model.setDT(Parameters::timestepMs);
    model.setName("s_mnist_test");
    model.setMergePostsynapticModels(true);
    model.setTiming(Parameters::timingEnabled);

    //---------------------------------------------------------------------------
    // Parameters and state variables
    //---------------------------------------------------------------------------
    // Recurrent ALIF population
    RecurrentALIF::ParamValues recurrentALIFParamVals(
        20.0,       // Membrane time constant [ms]
        2000.0,     // Adaption time constant [ms]
        0.6,        // Spiking threshold [mV]
        5.0,        // Refractory time constant [ms]
        0.0174);    // Scale of adaption [mV]

    RecurrentALIF::VarValues recurrentALIFInitVals(
        0.0,    // V
        0.0,    // A
        0.0,    // RefracTime
        0.0);   // E

    // Output population
    OutputClassification::ParamValues outputParamVals(20.0);    // Membrane time constant [ms]

    OutputClassification::VarValues outputInitVals(
        0.0,                    // Y
        0.0,                    // Pi
        uninitialisedVar());    // B

    //---------------------------------------------------------------------------
    // Neuron populations
    //---------------------------------------------------------------------------
    auto *input = model.addNeuronPopulation<InputSequential>("Input", Parameters::numInputNeurons,
                                                             {}, {});

    auto *recurrentALIF = model.addNeuronPopulation<RecurrentALIF>("RecurrentALIF", Parameters::numRecurrentNeurons,
                                                                   recurrentALIFParamVals, recurrentALIFInitVals);

    model.addNeuronPopulation<OutputClassification>("Output", Parameters::numOutputNeurons,
                                                    outputParamVals, outputInitVals);

#ifdef ENABLE_RECORDING
    input->setSpikeRecordingEnabled(true);
    recurrentALIF->setSpikeRecordingEnabled(true);
#endif

    //---------------------------------------------------------------------------
    // Synapse populations
    //---------------------------------------------------------------------------
    // Input->recurrent connections
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "InputRecurrentALIF", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Input", "RecurrentALIF",
        {}, {uninitialisedVar()},
        {}, {});

    // Recurrent->recurrent connections
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "ALIFALIFRecurrent", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "RecurrentALIF", "RecurrentALIF",
        {}, {uninitialisedVar()},
        {}, {});

    // Recurrent->output connections
    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "RecurrentALIFOutput", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "RecurrentALIF", "Output",
        {}, {uninitialisedVar()},
        {}, {});
}
