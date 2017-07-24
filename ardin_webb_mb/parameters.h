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
    constexpr double interStimuliDurationMs = 200.0;

    // Network dimensions
    constexpr unsigned int numPN = 360;
    constexpr unsigned int numKC = 20000;
    constexpr unsigned int numEN = 1;

    // Scale applied to convert normalised image data to input currents for PNs
    // **NOTE** manually tuned to get approximately 50% PN activation
    constexpr double inputCurrentScale = 0.335;

    // Weight of static synapses between PN and KC populations
    // **NOTE** manually tuend to get approximately 200/20000 KC firing sparsity
    constexpr double pnToKCWeight = 0.09;

    // Initial/maximum weight of plastic synapses between KC and EN populations
    constexpr double kcToENWeight = 0.015;

    // Time constant of dopamine
    constexpr double tauD = 20.0;

    // Scale of each dopamine 'spike'
    constexpr double dopamineStrength = 0.001;

    // How many PN neurons are connected to each KC
    constexpr unsigned int numPNSynapsesPerKC = 10;
}