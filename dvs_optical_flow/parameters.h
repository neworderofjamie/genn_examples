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

    const unsigned int inputSize = 640;
    const unsigned int kernelSize = 5;
    const unsigned int centreSize = 625;

    const unsigned int macroPixelSize = centreSize / kernelSize;

    const unsigned int detectorSize = macroPixelSize - 2;

    const unsigned int outputScale = 25;
    const unsigned int inputScale = 1;

    const float spikePersistence = 0.995f;

    const float outputVectorScale = 2.0f;
}
