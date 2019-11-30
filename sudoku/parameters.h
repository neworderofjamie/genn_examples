#pragma once

// Standard C++ includes
#include <string>

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

    inline std::string getPopName(size_t x, size_t y, size_t d)
    {
        return std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(d);
    }
}   // namespace Parameters