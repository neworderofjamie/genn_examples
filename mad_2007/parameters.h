#pragma once

// Standard C includes
#include <cmath>

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestep = 0.1;

    // number of cells
    const unsigned int numExcitatory = 90000;
    const unsigned int numInhibitory = 22500;

    // connection probability
    const double probabilityConnection = 0.1;
    
    const double excitatoryPeakWeight = 0.04561;
    
    const double excitatoryInhibitoryRatio = -5.0;
}