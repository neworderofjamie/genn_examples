#pragma once

namespace Parameters
{
    constexpr double timestepMs = 0.1;

    // Network structure
    constexpr unsigned int numInput = 200;
    constexpr unsigned int numOutput = 200;
    constexpr unsigned int numHidden = 256;

    // Model parameters
    constexpr double tauRise = 5.0;
    constexpr double tauDecay = 10.0;
    constexpr double tauRMS = 30000.0;
    constexpr double r0 = 0.001 * 1000.0;
    constexpr double epsilon = 1E-32;

    // Weights
    // **NOTE** Auryn units are volts, seconds etc so essentially 1000x GeNN parameters
    constexpr double wMin = -0.1 * 1000.0;
    constexpr double wMax = 0.1 * 1000.0;
    constexpr double w0 = 0.05 * 1000.0;

    // Experiment parameters
    constexpr double inputFreqHz = 5.0;
    constexpr unsigned int numTrials = 600;
    constexpr double updateTimeMs = 500.0;
    constexpr double trialMs = 1890.0;

    // Convert parameters to timesteps
    const unsigned long long updateTimesteps = (unsigned long long)(updateTimeMs / timestepMs);
    const unsigned int trialTimesteps = (unsigned int)(trialMs / timestepMs);
}
