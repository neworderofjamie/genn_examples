#pragma once

#include <random>

inline size_t ceilDivide(size_t numerator, size_t denominator)
{
    return ((numerator + denominator - 1) / denominator);
}

inline void buildRowLengths(unsigned int numPre, unsigned int numPost, unsigned int numThreadsPerSpike, size_t numConnections,
                            unsigned int *subRowLengths, std::mt19937 &rng)
{
    assert(numThreadsPerSpike > 0);

    // Calculate row lengths
    const size_t numPostPerThread = ceilDivide(numPost, numThreadsPerSpike);
    const size_t leftOverNeurons = numPost % numPostPerThread;

    size_t remainingConnections = numConnections;
    size_t matrixSize = (size_t)numPre * (size_t)numPost;

    // Loop through rows
    for(size_t i = 0; i < numPre; i++) {
        const bool lastPre = (i == (numPre - 1));

        // Loop through subrows
        for(size_t j = 0; j < numThreadsPerSpike; j++) {

            const bool lastSubRow = (j == (numThreadsPerSpike - 1));

            // If this isn't the last sub-row of the matrix
            if(!lastPre || ! lastSubRow) {
                // Get length of this subrow
                const unsigned int numSubRowNeurons = (leftOverNeurons != 0 && lastSubRow) ? leftOverNeurons : numPostPerThread;

                // Calculate probability
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
    }

    // Insert remaining connections into last sub-row
    *subRowLengths = (unsigned int)remainingConnections;
}
