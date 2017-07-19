#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestep = 1.0;

    // number of cells
    const unsigned int numExcitatory = 800;
    const unsigned int numInhibitory = 200;

    // connection probability
    const double probabilityConnection = 0.1;
}