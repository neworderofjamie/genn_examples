#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestep = 1.0;

    const unsigned int inputSize = 128;
    const unsigned int kernelSize = 5;
    const unsigned int centreSize = 45;

    const unsigned int macroPixelSize = centreSize / kernelSize;
}