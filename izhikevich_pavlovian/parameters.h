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
    constexpr double weightRecordIntervalMs = 10.0 * 1000.0;

    // STDP params
    constexpr double tauD = 200.0;

    // number of cells
    constexpr unsigned int numExcitatory = 800;
    constexpr unsigned int numInhibitory = 200;

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