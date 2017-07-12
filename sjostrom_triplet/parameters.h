#pragma once

// Standard C includes
#include <cmath>

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestep = 1.0;

    const double aScale = 0.5;

    const double frequencies[] = {0.1, 10.0, 20.0, 40.0, 50.0};
    const double dt[] = {-10.0, 10.0};

    const unsigned int numNeurons = (sizeof(frequencies) / sizeof(double)) * (sizeof(dt) / sizeof(double));

}