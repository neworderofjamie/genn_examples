#pragma once

//------------------------------------------------------------------------
// Parameters::Input
//------------------------------------------------------------------------
namespace Parameters
{
constexpr inline int ceilDivide(int numerator, int denominator)
{
    return ((numerator + denominator - 1) / denominator);
}

// Simulation timestep
constexpr double timestepMs = 0.1;

// Input layer neuron parameters
namespace Input
{
    // How long to present each stimuli for
    constexpr double presentMs = 100.0;

    // Scaling factor to multiply (0-255) input by to get rate
    constexpr double scale = 0.002;

    // Layer width
    constexpr int width = 44;

    // Layer height
    constexpr int height = 16;

    // Layer channels
    constexpr int channels = 1;

    // Number of neurons
    constexpr int numNeurons = width * height * channels;
}

//------------------------------------------------------------------------
// Parameters::InputConv1
//------------------------------------------------------------------------
namespace InputConv1
{
    // Number of filters
    constexpr int numFilters = 5;

    // Convolution size
    constexpr int convKernelWidth = 5;
    constexpr int convKernelHeight = 5;

    // Convolution stride
    constexpr int convStrideWidth = 1;
    constexpr int convStrideHeight = 1;

    // Total size of kernel
    constexpr int kernelSize = Input::channels * numFilters * convKernelWidth * convKernelHeight;
}

//------------------------------------------------------------------------
// Parameters::Conv1
//------------------------------------------------------------------------
namespace Conv1
{
    // WTA threshold
    constexpr double threshWTA = 8.0;

    // Inference threshold
    constexpr double threshInf = 40.0;

    // Radius of WTA in other features
    constexpr int WTARadius = 2;

    // Layer width
    constexpr int width = ceilDivide(Input::width - InputConv1::convKernelWidth + 1, InputConv1::convStrideWidth);

    // Layer height
    constexpr int height = ceilDivide(Input::height - InputConv1::convKernelHeight + 1, InputConv1::convStrideHeight);

    // Layer channels
    constexpr int channels = InputConv1::numFilters;

    // Number of neurons
    constexpr int numNeurons = width * height * channels;
}

//------------------------------------------------------------------------
// Parameters::Conv1Conv2
//------------------------------------------------------------------------
namespace Conv1Conv2
{
    // Pool size
    constexpr int poolKernelWidth = 2;
    constexpr int poolKernelHeight = 2;
    constexpr double poolScale = 1.0 / (double)(poolKernelWidth * poolKernelWidth);

    // Pool stride
    constexpr int poolStrideWidth = 2;
    constexpr int poolStrideHeight = 2;

    // Convolution input size
    constexpr int convInWidth = ceilDivide(Conv1::width - poolKernelWidth + 1, poolStrideWidth);
    constexpr int convInHeight = ceilDivide(Conv1::height - poolKernelHeight + 1, poolStrideHeight);

    // Number of filters
    constexpr int numFilters = 10;

    // Convolution size
    constexpr int convKernelWidth = 5;
    constexpr int convKernelHeight = 5;

    // Convolution stride
    constexpr int convStrideWidth = 1;
    constexpr int convStrideHeight = 1;

    // Total size of kernel
    constexpr int kernelSize = Conv1::channels * numFilters * convKernelWidth * convKernelHeight;
}

//------------------------------------------------------------------------
// Parameters::Conv2
//------------------------------------------------------------------------
namespace Conv2
{
    // WTA threshold
    constexpr double threshWTA = 30.0;

    // Inference threshold
    constexpr double threshInf = 30.0;

    // Radius of WTA in other features
    constexpr int WTARadius = 2;

    // Layer width
    constexpr int width = ceilDivide(Conv1Conv2::convInWidth - Conv1Conv2::convKernelWidth + 1, Conv1Conv2::convStrideWidth);

    // Layer height
    constexpr int height = ceilDivide(Conv1Conv2::convInHeight - Conv1Conv2::convKernelHeight + 1, Conv1Conv2::convStrideHeight);

    // Layer channels
    constexpr int channels = Conv1Conv2::numFilters;

    // Number of neurons
    constexpr int numNeurons = width * height * channels;
}

//------------------------------------------------------------------------
// Parameters::Conv1KC
//------------------------------------------------------------------------
namespace Conv1KC
{
    // How many Conv1 neurons are connected to each KC
    // **NOTE** this is after pooling - each of these synapses essentially connects to two presynaptic neurons
    constexpr unsigned int numSynapsesPerKC = 10;

    // Pool size
    constexpr int poolKernelWidth = 2;
    constexpr int poolKernelHeight = 2;
    constexpr double poolScale = 1.0 / (double)(poolKernelWidth * poolKernelWidth);
    
    // Weight of each synapse
    constexpr double weight = 1.0 / poolScale;
}

//------------------------------------------------------------------------
// Parameters::KC
//------------------------------------------------------------------------
namespace KC
{
    // Number of neurons
    constexpr int numNeurons = 20000;
    
    // WTA threshold
    constexpr double threshWTA = 40.0;
}

namespace GGN
{
    // Inference threshold
    constexpr double threshInf = 200.0;
}
}
