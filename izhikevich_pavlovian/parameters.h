#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    constexpr double timestepMs = 1.0;

    // Simulation duration
    constexpr double durationMs = 60.0 * 60.0 * 1000.0;

    // How much of start and end of simulation to record
    // **NOTE** we want to see at least one rewarded stimuli in each recording window
    constexpr double recordStartMs = 40.0 * 1000.0;
    constexpr double recordEndMs = 40.0 * 1000.0;

    // How often should outgoing weights from each synapse be recorded
    constexpr double weightRecordIntervalMs = durationMs;//10.0 * 1000.0;

    // STDP params
    constexpr double tauD = 200.0;

    // scaling
    constexpr unsigned int sizeScaleFactor = 1;
    constexpr double weightScaleFactor = 1.0 / (double)sizeScaleFactor;

    // scaled number of cells
    constexpr unsigned int numExcitatory = 800 * sizeScaleFactor;
    constexpr unsigned int numInhibitory = 200 * sizeScaleFactor;
    constexpr unsigned int numCells = numExcitatory + numInhibitory;

    // weights
    constexpr double inhWeight = -1.0 * weightScaleFactor;
    constexpr double initExcWeight = 1.0 * weightScaleFactor;
    constexpr double maxExcWeight = 4.0 * weightScaleFactor;
    constexpr double dopamineStrength = 0.5 * weightScaleFactor;

    // connection probability
    constexpr double probabilityConnection = 0.1;

    // input sets
    constexpr unsigned int numStimuliSets = 100;
    constexpr unsigned int stimuliSetSize = 50;
    constexpr double stimuliCurrent = 40.0;

    // regime
    constexpr double minInterStimuliIntervalMs = 100.0;
    constexpr double maxInterStimuliIntervalMs = 300.0;

    // reward
    constexpr double rewardDelayMs = 1000.0;
}