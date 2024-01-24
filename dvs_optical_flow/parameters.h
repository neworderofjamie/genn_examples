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

    constexpr unsigned int inputSize = 260;
    constexpr unsigned int kernelSize = 5;
    constexpr unsigned int centreSize = 245;

    constexpr unsigned int macroPixelSize = centreSize / kernelSize;

    constexpr unsigned int detectorSize = macroPixelSize - 2;

    constexpr unsigned int outputScale = 12;
    constexpr unsigned int inputScale = 1;

    constexpr float flowPersistence = 0.995f;
    constexpr float spikePersistence = 0.97f;
    
    constexpr float outputVectorScale = 2.0f;
}
