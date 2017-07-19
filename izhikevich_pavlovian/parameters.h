#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestepMs = 1.0;

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
}