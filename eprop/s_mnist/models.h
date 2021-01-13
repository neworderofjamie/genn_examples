#pragma once

//----------------------------------------------------------------------------
// InputSequential
//----------------------------------------------------------------------------
class InputSequential : public NeuronModels::Base
{
public:
    DECLARE_MODEL(InputSequential, 0, 0);
    
    SET_SIM_CODE(
        "const int globalTimestep = (int)$(t);\n"
        "const int trial = globalTimestep / ((28 * 28 * 2) + 20);\n"
        "const int timestep = globalTimestep % ((28 * 28 * 2) + 20);\n"
        "const uint8_t *imgData = &$(dataset)[$(indices)[trial] * 28 * 28];\n"
        "bool spike = false;\n"
        "// If we should be presenting the image\n"
        "if(timestep < (28 * 28 * 2)) {\n"
        "   const int mirroredTimestep = timestep / 2;\n"
            "if($(id) == 98) {\n"
        "       spike = (imgData[mirroredTimestep] == 255);\n"
        "   }\n"
        "   else if($(id) < 98 && mirroredTimestep < ((28 * 28) - 1)){\n"
        "       const int threshold = (int)((float)($(id) % 49) * (254.0 / 48.0));\n"
        "       // If this is an 'onset' neuron\n"
        "       if($(id) < 49) {\n"
        "           spike = ((imgData[mirroredTimestep] < threshold) && (imgData[mirroredTimestep + 1] >= threshold));\n"
        "       }\n"
        "       // If this is an 'offset' neuron\n"
        "       else {\n"
        "           spike = ((imgData[mirroredTimestep] >= threshold) && (imgData[mirroredTimestep + 1] < threshold));\n"
        "       }\n"
        "   }\n"
        "}\n"
        "// Otherwise, spike if this is the last 'touch' neuron\n"
        "else {\n"
        "   spike = ($(id) == 99);\n"
        "}\n");
    SET_THRESHOLD_CONDITION_CODE("spike");
    
    SET_EXTRA_GLOBAL_PARAMS({{"indices", "unsigned int*"}, {"dataset", "uint8_t*"}});
    
    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(InputSequential);