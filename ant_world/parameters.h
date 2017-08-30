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

    // Testing parameters
    constexpr double scanAngle = 120.0;
    constexpr double scanStep = 2.0;
    constexpr double spinStep = 0.5;
    constexpr double snapshotDistance = 10.0 / 100.0;
    constexpr double errorDistance = 20.0 / 100.0;

    // Network dimensions
    constexpr unsigned int inputWidth = 36;
    constexpr unsigned int inputHeight = 10;
    constexpr unsigned int numPN = inputWidth * inputHeight;
    constexpr unsigned int numKC = 20000;
    constexpr unsigned int numEN = 1;

    // Scale applied to convert image data to input currents for PNs
    // **NOTE** manually tuned to get approximately 50% PN activation
    constexpr double inputCurrentScale = 4.0;

    // Weight of static synapses between PN and KC populations
    // **NOTE** manually tuend to get approximately 200/20000 KC firing sparsity
    constexpr double pnToKCWeight = 0.0525;// 0.0705;

    // Initial/maximum weight of plastic synapses between KC and EN populations
    // **NOTE** note manually tuned to get 15-20 spikes for a novel image
    constexpr double kcToENWeight = 0.6;

    // Time constant of dopamine
    constexpr double tauD = 20.0;

    // Scale of each dopamine 'spike'
    // **NOTE** manually tuned for one-shot learning - also close to BA/phi
    constexpr double dopamineStrength = 0.03;

    // How many PN neurons are connected to each KC
    constexpr unsigned int numPNSynapsesPerKC = 10;

    // Rendering parameters
    // What colour should the ground be?
    constexpr GLfloat groundColour[] = {0.898f, 0.718f, 0.353f};

    // What colour should the brightest tussocks be?
    constexpr GLfloat worldColour[] = {0.0f, 1.0f, 0.0f};

    // Size of snapshots for initial pre-processing
    constexpr unsigned int intermediateSnapshotWidth = 74;
    constexpr unsigned int intermediateSnapshotHeight = 19;

    // How much larger than intermediate snapshots, rendering is performed at
    constexpr unsigned int displayScale = 8;

    // From these calculate display size
    constexpr unsigned int displayRenderWidth = intermediateSnapshotWidth * displayScale;
    constexpr unsigned int displayRenderHeight = intermediateSnapshotHeight * displayScale;

    // Ant parameters
    // How fast does the ant move?
    constexpr float antTurnSpeed = 4.0f;
    constexpr float antMoveSpeed = 0.05f;
}