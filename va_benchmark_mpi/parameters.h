#pragma once

// Standard C includes
#include <cmath>

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestep = 1.0;

    // number of cells
    const unsigned int numNeurons = 4000;

    const double resetVoltage = -60.0;
    const double thresholdVoltage = -50.0;

    // connection probability
    const double probabilityConnection = 0.1;

    // number of excitatory cells:number of inhibitory cells
    const double excitatoryInhibitoryRatio = 4.0;

    const unsigned int numExcitatory = (unsigned int)std::round(((double)numNeurons * excitatoryInhibitoryRatio) / (1.0 + excitatoryInhibitoryRatio));
    const unsigned int numInhibitory = numNeurons - numExcitatory;

    const double scale = (4000.0 / (double)numNeurons) * (0.02 / probabilityConnection);

    const double excitatoryWeight = 4.0E-3 * scale;
    const double inhibitoryWeight = -51.0E-3 * scale;

}