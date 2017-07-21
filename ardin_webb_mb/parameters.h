#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    constexpr double timestepMs = 1.0;

    // Simulation duration
    constexpr double durationMs = 50.0;

    // Regime parameters
    constexpr double rewardTimeMs = 40.0;
    constexpr double presentDurationMs = 40.0;

    // Network dimensions
    constexpr unsigned int numPN = 360;
    constexpr unsigned int numKC = 20000;
    constexpr unsigned int numEN = 1;

    // Learning parameters
    constexpr double tauD = 20.0;

     // **HACK** 0.1 is a random scaling factor
    constexpr double weightScale = 0.1;

    // How many PN neurons are connected to each KC
    constexpr unsigned int numPNSynapsesPerKC = 10;
}