#pragma once

// Standard C++ includes
#include <string>

//---------------------------------------------------------------------
// Parameters
//---------------------------------------------------------------------
namespace Parameters
{
    // Number of neurons per variables (note < blockzise is a bit wasteful)
    constexpr unsigned int coreSize = 25;
    
    // Runtime of simulation
    constexpr double runTimeMs = 60000.0;
    
    constexpr unsigned int delay = 1;

    inline std::string getPopName(size_t x, size_t y)
    {
        return std::to_string(x) + "_" + std::to_string(y);
    }
}   // namespace Parameters
