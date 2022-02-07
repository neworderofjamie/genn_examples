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

    constexpr double timestep = 1.0;

    constexpr unsigned int inputSize = 256;
    constexpr unsigned int kernelSize = 5;
    constexpr unsigned int centreSize = 250;

    constexpr unsigned int macroPixelSize = centreSize / kernelSize;

    constexpr unsigned int detectorSize = macroPixelSize - 2;

    constexpr unsigned int outputScale = 10;
    constexpr unsigned int inputScale = 2;

    constexpr float spikePersistence = 0.995f;
    
    constexpr float outputVectorScale = 2.0f;
}
