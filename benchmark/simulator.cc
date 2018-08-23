#include <random>

// GeNN robotics includes
#include "common/timer.h"
#include "genn_utils/connectors.h"

#include "parameters.h"

#include "benchmark_CODE/definitions.h"

using namespace BoBRobotics;

int main()
{
    {
        Timer<> t("Allocation:");
        allocateMem();
    }

    {
        Timer<> t("Calculating row lengths:");

        std::random_device rd;
        std::mt19937 gen(rd());
        
        unsigned int *rowLengths = nullptr;
#ifndef CPU_ONLY
        CHECK_CUDA_ERRORS(cudaMallocHost(&rowLengths, Parameters::numNeurons * sizeof(unsigned int)));
#else
        initSparseConnrowLengthSyn = new unsigned int[Parameters::numNeurons];
        rowLengths = initSparseConnrowLengthSyn;
#endif
        // Calculate row lengths
        // **NOTE** we are FINISHING at second from last row because all remaining connections must go in last row
        size_t remainingConnections = Parameters::numConnections;
        size_t matrixSize = (size_t)Parameters::numNeurons * (size_t)Parameters::numNeurons;
        std::generate_n(&rowLengths[0], Parameters::numNeurons - 1,
                        [&remainingConnections, &matrixSize, &gen]()
                        {
                            const double probability = (double)Parameters::numNeurons / (double)matrixSize;

                            // Create distribution to sample row length
                            std::binomial_distribution<size_t> rowLengthDist(remainingConnections, probability);

                            // Sample row length;
                            const size_t rowLength = rowLengthDist(gen);

                            // Update counters
                            remainingConnections -= rowLength;
                            matrixSize -= Parameters::numNeurons;

                            return (unsigned int)rowLength;
                        });

        // Insert remaining connections into last row
        rowLengths[Parameters::numNeurons - 1] = (unsigned int)remainingConnections;

#ifndef CPU_ONLY
        CHECK_CUDA_ERRORS(cudaMalloc(&initSparseConnrowLengthSyn, Parameters::numNeurons * sizeof(unsigned int)));

        CHECK_CUDA_ERRORS(cudaMemcpy(initSparseConnrowLengthSyn, rowLengths, Parameters::numNeurons * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERRORS(cudaFreeHost(rowLengths));
#endif
    }

    {
        Timer<> t("Initialization:");
        initialize();
    }
    std::cout << "\tHost:" << initHost_tme * 1000.0 << std::endl;
    std::cout << "\tDevice:" << initDevice_tme * 1000.0 << std::endl;


    // Final setup
    {
        Timer<> t("Sparse init:");
        // Perform sparse initialisation
        initbenchmark();
    }
    std::cout << "\tHost:" << sparseInitHost_tme * 1000.0 << std::endl;
    std::cout << "\tDevice:" << sparseInitDevice_tme * 1000.0 << std::endl;

    // **HACK** Download row lengths and indices
    extern unsigned int *d_rowLengthSyn;
    extern unsigned int *d_indSyn;
    printf("%p, %p\n", CSyn.rowLength, d_rowLengthSyn);
    CHECK_CUDA_ERRORS(cudaMemcpy(CSyn.rowLength, d_rowLengthSyn, Parameters::numNeurons * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(CSyn.ind, d_indSyn, Parameters::numNeurons * CSyn.maxRowLength * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::cout << "Total connections:" << std::accumulate(&CSyn.rowLength[0], &CSyn.rowLength[Parameters::numNeurons], 0) << "(" << Parameters::numConnections << ")" << std::endl;

    std::vector<unsigned int> histogram(Parameters::numNeurons);
    for(unsigned int i = 0; i < Parameters::numNeurons; i++) {
        const unsigned int *rowInd = &CSyn.ind[i * CSyn.maxRowLength];
        assert(std::is_sorted(&rowInd[0], &rowInd[CSyn.rowLength[i]]));

        for(unsigned int j = 0; j < CSyn.rowLength[i]; j++) {
            const unsigned int s = CSyn.ind[(i * CSyn.maxRowLength) + j];
            assert(s < Parameters::numNeurons);
            histogram[s]++;
        }
    }

    for(unsigned int h : histogram) {
        std::cout << h << ", ";
    }
    std::cout << std::endl;
  return 0;
}
