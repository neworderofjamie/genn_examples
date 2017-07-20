#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    constexpr double timestepMs = 1.0;

    // Network dimensions
    constexpr unsigned int numPN = 360;
    constexpr unsigned int numKC = 20000;
    constexpr unsigned int numEN = 1;

    // Learning parameters
    constexpr double tauD = 20.0;

    // How many PN neurons are connected to each KC
    constexpr unsigned int numPNSynapsesPerKC = 10;
}