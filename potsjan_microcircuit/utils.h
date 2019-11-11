#pragma once

#include <random>

inline size_t ceilDivide(size_t numerator, size_t denominator)
{
    return ((numerator + denominator - 1) / denominator);
}

inline void buildRowLengths(unsigned int numPre, unsigned int numPost, unsigned int numThreadsPerSpike, size_t numConnections,
                            unsigned int *subRowLengths, std::mt19937 &rng)
{
    // Calculate row lengths
    // **NOTE** we are FINISHING at second from last row because all remaining connections must go in last row
    //const size_t numSubRows = numPre * numThreadsPerSpike;
    const size_t numPostPerThread = ceilDivide(numPost, numThreadsPerSpike);
    const size_t leftOverNeurons = numPost % numPostPerThread;

    size_t remainingConnections = numConnections;
    size_t matrixSize = (size_t)numPre * (size_t)numPost;

    for(size_t i = 0; i < numPre; i++) {
        for(size_t j = 0; j < numThreadsPerSpike; j++) {
            // Get length of this subrow
            const bool lastSubRow = (j == (numThreadsPerSpike - 1));
            const unsigned int numSubRowNeurons = (leftOverNeurons != 0 && lastSubRow) ? leftOverNeurons : numPostPerThread;

            const double probability = (double)numSubRowNeurons / (double)matrixSize;

            // Create distribution to sample row length
            std::binomial_distribution<size_t> rowLengthDist(remainingConnections, probability);

            // Sample row length;
            const size_t subRowLength = rowLengthDist(rng);

            // Update counters
            remainingConnections -= subRowLength;
            matrixSize -= numSubRowNeurons;

            // Add row length to array
            *subRowLengths++ = (unsigned int)subRowLength;
        }
    }

    // Insert remaining connections into last row
    *subRowLengths = (unsigned int)remainingConnections;
}
