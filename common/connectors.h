#pragma once

// Standard C++ includes
#include <algorithm>
#include <stdexcept>
#include <vector>

// Standard C includes
#include <cassert>
#include <cmath>

// GeNN includes
#include "sparseProjection.h"

//----------------------------------------------------------------------------
// Typedefines
//----------------------------------------------------------------------------
typedef void (*AllocateFn)(unsigned int);

//----------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------
// Calculate log factorial using lookup table and log gamma function from standard library
// Adopted from numerical recipes in C p170
inline double lnFact(int n)
{
    // **NOTE** a static array is automatically initialized  to zero.
    static double a[101];
    if (n < 0) {
        throw std::runtime_error("Negative factorial in routine factln");
    }
    else if (n <= 1) {
        return 0.0;
    }
    // In range of table.
    else if (n <= 100) {
        return a[n] ? a[n] : (a[n] = lgamma(n + 1.0));
    }
    // Out of range of table.
    else {
        return lgamma(n + 1.0);
    }
}
//----------------------------------------------------------------------------
// Calculate natural log of binomial coefficient using log factorial
// Adopted from numerical recipes in C p169
inline double lnBinomialCoefficient(int n, int k)
{
    return lnFact(n) - lnFact(k) - lnFact(n - k);
}
//----------------------------------------------------------------------------
// Evaluates PDF of binomial distribution
// Adopted from C++ 'prob' libray found https://people.sc.fsu.edu/~jburkardt/
inline double binomialPDF(int n, int k, double p)
{
    if(n < 1) {
        return 0.0;
    }
    else if(k < 0 || n < k) {
        return 0.0;
    }
    else if(p == 0.0) {
        if(k == 0) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }
    else if(p == 1.0) {
        if(k == n) {
           return 1.0;
        }
        else {
            return 0.0;
        }
    }
    else {
        return exp(lnBinomialCoefficient(n, k) + (k * log(p)) + ((n - k) * log(1.0 - p)));
    }
}
//----------------------------------------------------------------------------
// Evaluates inverse CDF of binomial distribution
// Adopted from C++ 'prob' libray found https://people.sc.fsu.edu/~jburkardt/
inline unsigned int binomialInverseCDF(double cdf, unsigned int n, double p)
{
    if(cdf < 0.0 || 1.0 < cdf) {
        throw std::runtime_error("binomialInverseCDF error - CDF < 0 or 1 < CDF");
    }

    double cdf2 = 0.0;
    for (unsigned int k = 0; k <= n; k++)
    {
        const double pdf = binomialPDF(n, k, p);
        cdf2 += pdf;

        if (cdf2 > cdf) {
            return k;
        }

    }

    throw std::runtime_error("Invalid CDF parameterse");
}
//----------------------------------------------------------------------------
inline void addSynapseToSparseProjection(unsigned int i, unsigned int j, unsigned int numPre,
                                         SparseProjection &sparseProjection)
{
    // Get index of current end of row in sparse projection
    const unsigned int rowEndIndex = sparseProjection.indInG[i + 1];

    // Also get index of last synapse
    const unsigned int lastSynapseIndex = sparseProjection.indInG[numPre];

    // If these aren't the same (there are existing synapses after this one), shuffle up the indices
    if(rowEndIndex != lastSynapseIndex) {
        std::move_backward(&sparseProjection.ind[rowEndIndex], &sparseProjection.ind[lastSynapseIndex],
                            &sparseProjection.ind[lastSynapseIndex + 1]);
    }

    // Insert new synapse
    sparseProjection.ind[rowEndIndex] = j;

    // Increment all subsequent indices
    std::transform(&sparseProjection.indInG[i + 1], &sparseProjection.indInG[numPre + 1], &sparseProjection.indInG[i + 1],
                   [](unsigned int index)
                   {
                       return (index + 1);
                   });
}

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
//----------------------------------------------------------------------------
unsigned int calcFixedProbabilityConnectorMaxConnections(unsigned int numPre, unsigned int numPost, double probability)
{
    // Calculate suitable quantile for 0.9999 change when drawing numPre times
    const double quantile = pow(0.9999, 1.0 / (double)numPre);

    return binomialInverseCDF(quantile, numPost, probability);
}

//----------------------------------------------------------------------------
template <typename Generator>
void buildFixedNumberPreConnector(unsigned int numPre, unsigned int numPost, unsigned int numConnections,
                                  SparseProjection &projection, AllocateFn allocate, Generator &gen)
{
    // Allocate sparse projection
    allocate(numPost * numConnections);

    // Zero all indInG
    std::fill(&projection.indInG[0], &projection.indInG[numPre + 1], 0);

    // Generate array of presynaptic indices
    std::vector<unsigned int> preIndices(numPre);
    std::iota(preIndices.begin(), preIndices.end(), 0);

    // Loop through postsynaptic neurons
    for(unsigned int j = 0; j < numPost; j++) {
        // Loop through connections to make
        for(unsigned int c = 1; c <= numConnections; c++) {
            // Create distribution to select from remaining available neurons
            std::uniform_int_distribution<> dis(0, numPre - c);

            // Pick a presynaptic neuron
            const unsigned int i = preIndices[dis(gen)];

            // Add synapse to projection
            addSynapseToSparseProjection(i, j, numPre, projection);

            // Swap the last available preindex with the one we have now used
            std::swap(preIndices[i], preIndices[numPre - c]);
        }
    }

    // Check correct number of connections were added
    assert(projection.indInG[numPre] == projection.connN);
}
//----------------------------------------------------------------------------
unsigned int calcFixedNumberPreConnectorMaxConnections(unsigned int numPre, unsigned int numPost, unsigned int numConnections)
{
    // Calculate suitable quantile for 0.9999 change when drawing numPre times
    const double quantile = pow(0.9999, 1.0 / (double)numPre);

    return binomialInverseCDF(quantile, numPost, (double)numConnections / (double)numPre);
}