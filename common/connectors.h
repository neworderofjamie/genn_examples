#pragma once

// Standard C++ includes
#include <algorithm>
#include <stdexcept>
#include <vector>

// Standard C includes
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
double lnFact(int n)
{
    // **NOTE** a static array is automatically initialized  to zero.
    static double a[101];

    if (n < 0) {
        throw std::runtime_error("Negative factorial in routine factln");
    }
    else if (n <= 1) {
        return 0.0;
    }
    //In range of table.
    else if (n <= 100) {
        return a[n] ? a[n] : (a[n] = lgamma(n + 1.0));
    }
    // Out  of range  of  table.
    else {
        return lgamma(n + 1.0);
    }
}
//----------------------------------------------------------------------------
// Calculate binomial coefficient using log factorial
// Adopted from numerical recipes in C p169
double binomialCoefficient(int n, int k)
{
    return floor(0.5 + exp(lnFact(n) - lnFact(k) - lnFact(n - k)));
}
//----------------------------------------------------------------------------
// Evaluates PDF of binomial distribution
// Adopted from C++ 'prob' libray found https://people.sc.fsu.edu/~jburkardt/
double binomialPDF(int x, int a, double b)
{
    if(a < 1) {
        return 0.0;
    }
    else if(x < 0 || a < x) {
        return 0.0;
    }
    else if(b == 0.0) {
        if(x == 0) {
            return 1.0;
        }
        else {
            return 0.0;
        }
    }
    else if(b == 1.0) {
        if(x == a) {
           return 1.0;
        }
        else {
            return 0.0;
        }
    }
    else {
        return binomialCoefficient(a, x) * pow(b, x) * pow(1.0 - b, a - x);
    }
}
//----------------------------------------------------------------------------
// Evaluates inverse CDF of binomial distribution
// Adopted from C++ 'prob' libray found https://people.sc.fsu.edu/~jburkardt/
unsigned int binomialInverseCDF(double cdf, unsigned int a, double b)
{
    if(cdf < 0.0 || 1.0 < cdf) {
        throw std::runtime_error("binomialInverseCDF error - CDF < 0 or 1 < CDF");
    }

    double cdf2 = 0.0;
    for (unsigned int x = 0; x <= a; x++)
    {
        const double pdf = binomialPDF(x, a, b);
        cdf2 += pdf;

        if (cdf <= cdf2) {
            return x;
        }

    }

    throw std::runtime_error("Invalid CDF parameterse");
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
unsigned int calcFixedProbabilityConnectorMaxConnections(unsigned int numPre, unsigned int numPost, float probability)
{
    // Calculate suitable quantile for 0.9999 change when drawing numPre times
    const float quantile = pow(0.9999, 1.0 / (float)numPre);

    return binomialInverseCDF(quantile, numPost, probability);
}