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
    const double recordStartMs = 50.0 * 1000.0;
    const double recordEndMs = 50.0 * 1000.0;

    // How often should outgoing weights from each synapse be recorded
    const double weightRecordIntervalMs = durationMs;//10.0 * 1000.0;

    // STDP params
    const double tauD = 200.0;

    // scaling
    const unsigned int sizeScaleFactor = 1;
    const double weightScaleFactor = 1.0 / (double)sizeScaleFactor;

    // scaled number of cells
    const unsigned int numExcitatory = 800 * sizeScaleFactor;
    const unsigned int numInhibitory = 200 * sizeScaleFactor;
    const unsigned int numCells = numExcitatory + numInhibitory;

    // weights
    const double inhWeight = -1.0 * weightScaleFactor;
    const double initExcWeight = 1.0 * weightScaleFactor;
    const double maxExcWeight = 4.0 * weightScaleFactor;
    const double dopamineStrength = 0.5 * weightScaleFactor;

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
    
    const bool measureTiming = false;
}
