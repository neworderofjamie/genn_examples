#pragma once

//---------------------------------------------------------------------
// Parameters
//---------------------------------------------------------------------
namespace Parameters
{
    // Number of neurons per variables (note < blockzise is a bit wasteful)
    const unsigned int coreSize = 25;
    
    // Runtime of simulation
    const double runTimeMs = 60000.0;
    
    const unsigned int delay = 0;
}   // namespace Parameters