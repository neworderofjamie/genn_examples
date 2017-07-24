#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    constexpr double timestepMs = 1.0;

    // Regime parameters
    constexpr double rewardTimeMs = 40.0;
    constexpr double presentDurationMs = 40.0;
    constexpr double interStimuliDurationMs = 100.0;

    // Network dimensions
    constexpr unsigned int numPN = 360;
    constexpr unsigned int numKC = 20000;
    constexpr unsigned int numEN = 1;

    // Learning parameters
    constexpr double tauD = 20.0;

    // **HACK** these are manually tuned to match dynamics in fig 3
    constexpr double pnToKCWeightScale = 0.07;
    constexpr double kcToENWeightScale = 0.005;

    // How many PN neurons are connected to each KC
    constexpr unsigned int numPNSynapsesPerKC = 10;
}