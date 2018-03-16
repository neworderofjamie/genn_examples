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
    const unsigned int numEEConnections = (unsigned int)std::round(probabilityConnection * (double)numExcitatory * (double)numExcitatory);
    const unsigned int numEIConnections = (unsigned int)std::round(probabilityConnection * (double)numExcitatory * (double)numInhibitory);
    const unsigned int numIEConnections = (unsigned int)std::round(probabilityConnection * (double)numInhibitory * (double)numExcitatory);
    const unsigned int numIIConnections = (unsigned int)std::round(probabilityConnection * (double)numInhibitory * (double)numInhibitory);

    const double excitatoryPeakWeight = 0.04561;
    
    const double externalInputRate = (9000.0 * 2.32);
    const double excitatoryInhibitoryRatio = -5.0;
}