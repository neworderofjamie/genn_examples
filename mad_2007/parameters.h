#pragma once

// Standard C includes
#include <cmath>

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const bool smallScale = false;

    const double timestep = 0.1;

    // number of cells
    const unsigned int numExcitatory = smallScale ? 900 : 90000;
    const unsigned int numInhibitory = smallScale ? 225 : 22500;

    // connection probability
    const double probabilityConnection = 0.1;
    
    const double excitatoryPeakWeight = smallScale ? 0.18244 : 0.04561;
    
    const double externalInputRate = smallScale ? (90.0 * 34.66) : (9000.0 * 2.32);
    const double excitatoryInhibitoryRatio = smallScale ? -18.0 : -5.0;
}