#pragma once

//#define ENABLE_RECORDING
//#define RESUME_EPOCH 0

namespace Parameters
{
    constexpr double timestepMs = 1.0;
    constexpr bool timingEnabled = true;

    constexpr unsigned int inputWidth = 28;
    constexpr unsigned int inputHeight = 28;
    constexpr unsigned int inputRepeats = 2;
    constexpr unsigned int cueDuration = 20;
    constexpr unsigned int trialTimesteps = (inputWidth * inputHeight * inputRepeats) + cueDuration;

    constexpr unsigned int batchSize = 512;

    constexpr unsigned int numInputNeurons = 100;
    constexpr unsigned int numRecurrentNeurons = 800;
    constexpr unsigned int numOutputNeurons = 16;
    
    constexpr double adamBeta1 = 0.9;
    constexpr double adamBeta2 = 0.999;
}
