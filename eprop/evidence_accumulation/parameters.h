#pragma once

namespace Parameters
{
    constexpr double timestepMs = 1.0;
    constexpr bool timingEnabled = true;

    constexpr unsigned int numInputPopulations = 4;
    constexpr unsigned int numInputPopulationNeurons = 10;
    constexpr unsigned int numInputNeurons = numInputPopulations * numInputPopulationNeurons;
    constexpr unsigned int numRecurrentNeurons = 600;
    constexpr unsigned int numOutputNeurons = 3;

    
    // Learning procedure parameters
    constexpr double activeRateHz = 40.0;
    constexpr double inactiveRateHz = 0.00000001;
    constexpr double backgroundRateHz = 10.0;

    constexpr double cuePresentMs = 100.0;
    constexpr double cueDelayMs = 50.0;

    constexpr double minDelayMs = 500.0;
    constexpr double maxDelayMs = 1000.0;

    constexpr double decisionMs = 150.0;

    constexpr unsigned int cuePresentTimesteps = (unsigned int)(cuePresentMs / timestepMs);
    constexpr unsigned int cueDelayTimesteps = (unsigned int)(cueDelayMs / timestepMs);
    constexpr unsigned int minDelayTimesteps = (unsigned int)(minDelayMs / timestepMs);
    constexpr unsigned int maxDelayTimesteps = (unsigned int)(maxDelayMs / timestepMs);
    constexpr unsigned int decisionTimesteps = (unsigned int)(decisionMs / timestepMs);
}