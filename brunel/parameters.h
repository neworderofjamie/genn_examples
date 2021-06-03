#pragma once

// Standard C includes
#include <cmath>

// Toggle STDP
//#define STDP
//#define SLOW_POISSON

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestep = 0.1;

    const double resetVoltage = 0.0;
    const double thresholdVoltage = 20.0;

    // Number of cells
    const unsigned int numNeurons = 10000;

    const unsigned int numTimesteps = 10000;

    // connection probability
    const double probabilityConnection = 0.1;

    // number of excitatory cells:number of inhibitory cells
    const double excitatoryInhibitoryRatio = 4.0;

    // Rate of Poisson noise injected into each neuron (Hz)
    const double inputRate = 20.0;

    const unsigned int numExcitatory = (unsigned int)std::round(((double)numNeurons * excitatoryInhibitoryRatio) / (1.0 + excitatoryInhibitoryRatio));
    const unsigned int numInhibitory = numNeurons - numExcitatory;

    const double scale = (10000.0 / (double)numNeurons) * (0.1 / probabilityConnection);

    const double excitatoryWeight = 0.1 * scale;
    const double inhibitoryWeight = -0.5 * scale;

    // Axonal delay
    const double delayMs = 1.5;

    const unsigned int delayTimesteps = (unsigned int)std::round(delayMs / timestep);

}
