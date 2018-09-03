#pragma once

// Standard C includes
#include <cmath>

#define MEASURE_TIMING
//#define STATIC

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestep = 0.1;

    const double durationMs = 200.0 * 1000.0;

    const double delayMs = 1.5;

    const unsigned int delayTimestep = (unsigned int)(delayMs / timestep) - 1;

    // number of cells
    const unsigned int numExcitatory = 90000;
    const unsigned int numInhibitory = 22500;

    // connection probability
    const double probabilityConnection = 0.1;

    const double excitatoryPeakWeight = 0.04561;
    
    const double externalInputRate = (9000.0 * 2.32);
    const double excitatoryInhibitoryRatio = -5.0;
}
