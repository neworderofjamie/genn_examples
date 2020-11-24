#pragma once

namespace Parameters
{
    constexpr double timestepMs = 1.0;
    constexpr bool timingEnabled = true;

    constexpr unsigned int numInputPopulations = 4;
    constexpr unsigned int numInputPopulationNeurons = 10;
    constexpr unsigned int numInputNeurons = numInputPopulations * numInputPopulationNeurons;
    constexpr unsigned int numRecurrentNeurons = 50;
    constexpr unsigned int numOutputNeurons = 2;

    
    // Learning procedure parameters
    constexpr float activeRateHz = 40.0f;
    constexpr float inactiveRateHz = 0.00000001f;
    constexpr float backgroundRateHz = 10.0f;

    constexpr double cuePresentMs = 100.0;
    constexpr double cueDelayMs = 50.0;

    constexpr double minDelayMs = 500.0;
    constexpr double maxDelayMs = 1500.0;

    constexpr double decisionMs = 150.0;

    constexpr unsigned int cuePresentTimesteps = (unsigned int)(cuePresentMs / timestepMs);
    constexpr unsigned int cueDelayTimesteps = (unsigned int)(cueDelayMs / timestepMs);
    constexpr unsigned int minDelayTimesteps = (unsigned int)(minDelayMs / timestepMs);
    constexpr unsigned int maxDelayTimesteps = (unsigned int)(maxDelayMs / timestepMs);
    constexpr unsigned int decisionTimesteps = (unsigned int)(decisionMs / timestepMs);
}