#pragma once

namespace Parameters
{
    constexpr double timestepMs = 1.0;
    constexpr bool timingEnabled = true;

    constexpr unsigned int inputWidth = 28;
    constexpr unsigned int inputHeight = 28;
    constexpr unsigned int inputRepeats = 2;
    constexpr unsigned int cueDuration = 20;
    constexpr unsigned int trialTimesteps = (inputWidth * inputHeight * inputRepeats) + cueDuration;

    constexpr unsigned int batchSize = 64;

    constexpr unsigned int numInputNeurons = 100;
    constexpr unsigned int numRecurrentNeurons = 400;
    constexpr unsigned int numOutputNeurons = 16;
}