#pragma once

//#define USE_DEEP_R

namespace Parameters
{
    constexpr bool timingEnabled = true;

    constexpr unsigned int numInputNeurons = 20;
    constexpr unsigned int numRecurrentNeurons = 600;
    constexpr unsigned int numOutputNeurons = 3;
    
    constexpr double deepRRecurrentConnectivity = 0.1;
}