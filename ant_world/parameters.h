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
    constexpr double postStimuliDurationMs = 200.0;

    // Network dimensions
    constexpr unsigned int inputWidth = 36;
    constexpr unsigned int inputHeight = 10;
    constexpr unsigned int numPN = inputWidth * inputHeight;
    constexpr unsigned int numKC = 20000;
    constexpr unsigned int numEN = 1;

    // Scale applied to convert image data to input currents for PNs
    // **NOTE** manually tuned to get approximately 50% PN activation
    constexpr double inputCurrentScale = 0.0115;

    // Weight of static synapses between PN and KC populations
    // **NOTE** manually tuend to get approximately 200/20000 KC firing sparsity
    constexpr double pnToKCWeight = 0.075;

    // Initial/maximum weight of plastic synapses between KC and EN populations
    // **NOTE** note manually tuned to get about 14 spikes for a novel image
    constexpr double kcToENWeight = 0.2;

    // Time constant of dopamine
    constexpr double tauD = 20.0;

    // Scale of each dopamine 'spike'
    // **NOTE** manually tuned for one-shot learning
    constexpr double dopamineStrength = 0.005;

    // How many PN neurons are connected to each KC
    constexpr unsigned int numPNSynapsesPerKC = 10;
}