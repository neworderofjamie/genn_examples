#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    // Order of detectors associated with each pixel
    enum Detector
    {
        DetectorLeft,
        DetectorRight,
        DetectorUp,
        DetectorDown,
        DetectorMax,
    };

    const double timestep = 1.0;

    const unsigned int inputSize = 128;
    const unsigned int kernelSize = 5;
    const unsigned int centreSize = 45;

    const unsigned int macroPixelSize = centreSize / kernelSize;

    const unsigned int detectorSize = macroPixelSize - 2;
}