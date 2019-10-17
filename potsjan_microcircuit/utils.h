#pragma once

#include <random>

void buildRowLengths(unsigned int numPre, unsigned int numPost, size_t numConnections, unsigned int *rowLengths, std::mt19937 &rng)
{
    // Calculate row lengths
    // **NOTE** we are FINISHING at second from last row because all remaining connections must go in last row
    size_t remainingConnections = numConnections;
    size_t matrixSize = (size_t)numPre * (size_t)numPost;
    std::generate_n(&rowLengths[0], numPre - 1,
                    [&remainingConnections, &matrixSize, numPost, &rng]()
                    {
                        const double probability = (double)numPost / (double)matrixSize;

                        // Create distribution to sample row length
                        std::binomial_distribution<size_t> rowLengthDist(remainingConnections, probability);

                        // Sample row length;
                        const size_t rowLength = rowLengthDist(rng);

                        // Update counters
                        remainingConnections -= rowLength;
                        matrixSize -= numPost;

                        return (unsigned int)rowLength;
                    });

    // Insert remaining connections into last row
    rowLengths[numPre - 1] = (unsigned int)remainingConnections;
}
