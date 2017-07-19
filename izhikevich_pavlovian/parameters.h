#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestepMs = 1.0;

    // Simulation duration
    const double durationMs = 60.0 * 60.0 * 1000.0;

    // How much of start and end of simulation to record
    // **NOTE** we want to see at least one rewarded stimuli in each recording window
    const double recordStartMs = 40.0 * 1000.0;
    const double recordEndMs = 40.0 * 1000.0;

    // STDP params
    const double tauD = 200.0;

    // number of cells
    const unsigned int numExcitatory = 800;
    const unsigned int numInhibitory = 200;

    // connection probability
    const double probabilityConnection = 0.1;

    // input sets
    const unsigned int numStimuliSets = 100;
    const unsigned int stimuliSetSize = 50;
    const double stimuliCurrent = 40.0;

    // regime
    const double minInterStimuliIntervalMs = 100.0;
    const double maxInterStimuliIntervalMs = 300.0;

    // reward
    const double rewardDelayMs = 1000.0;
}