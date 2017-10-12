#pragma once


//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    // Population sizes
    const unsigned int numTN2 = 2;
    const unsigned int numTL = 16;
    const unsigned int numCL1 = 16;
    const unsigned int numTB1 = 8;
    const unsigned int numCPU4 = 16;
    const unsigned int numPontine = 16;
    const unsigned int numCPU1 = 16;

    const double c = 0.33;

    enum Hemisphere
    {
        HemisphereLeft,
        HemisphereRight,
        HemisphereMax,
    };
}