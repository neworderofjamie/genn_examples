#pragma once

// Standard C++ includes
#include <algorithm>
#include <vector>

// GeNN includes
#include "sparseProjection.h"

//----------------------------------------------------------------------------
// Typedefines
//----------------------------------------------------------------------------
typedef void (*AllocateFn)(unsigned int);

//----------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------
template <typename Generator>
void buildFixedProbabilityConnector(unsigned int numPre, unsigned int numPost, float probability,
                                    SparseProjection &projection, AllocateFn allocate, Generator &gen)
{
  // Allocate memory for indices
  // **NOTE** RESIZE as this vector is populated by index
  std::vector<unsigned int> tempIndInG;
  tempIndInG.resize(numPre + 1);

  // Reserve a temporary vector to store indices
  std::vector<unsigned int> tempInd;
  tempInd.reserve((unsigned int)((float)(numPre * numPost) * probability));

  // Create RNG to draw probabilities
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // Loop through pre neurons
  for(unsigned int i = 0; i < numPre; i++)
  {
    // Connections from this neuron start at current end of indices
    tempIndInG[i] = tempInd.size();

    // Loop through post neurons
    for(unsigned int j = 0; j < numPost; j++)
    {
      // If there should be a connection here, add one to temporary array
      if(dis(gen) < probability)
      {
        tempInd.push_back(j);
      }
    }
  }

  // Add final index
  tempIndInG[numPre] = tempInd.size();

  // Allocate SparseProjection arrays
  // **NOTE** shouldn't do directly as underneath it may use CUDA or host functions
  allocate(tempInd.size());

  // Copy indices
  std::copy(tempIndInG.begin(), tempIndInG.end(), &projection.indInG[0]);
  std::copy(tempInd.begin(), tempInd.end(), &projection.ind[0]);
}