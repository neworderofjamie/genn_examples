#include <algorithm>
#include <chrono>
#include <random>

#include "model.cc"
#include "benchmark_CODE/definitions.h"

typedef void (*allocateFn)(unsigned int);

template<typename Generator>
void build_fixed_probability_connector(unsigned int numPre, unsigned int numPost, float probability,
                                       SparseProjection &projection, allocateFn allocate, Generator &gen)
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

int main()
{
    auto  allocStart = chrono::steady_clock::now();
    allocateMem();
    auto  allocEnd = chrono::steady_clock::now();
    printf("Allocation %ldms\n", chrono::duration_cast<chrono::milliseconds>(allocEnd - allocStart).count());

    auto  initStart = chrono::steady_clock::now();
    initialize();

    std::random_device rd;
    std::mt19937 gen(rd());

    build_fixed_probability_connector(NUM_PRE_NEURONS, NUM_POST_NEURONS, 0.1f,
                                      CSyn, &allocateSyn, gen);

    // Copy conductances
    std::fill(&gSyn[0], &gSyn[CSyn.connN], 0.0);

    // Convert input rate into a RNG threshold and fill
    float inputRate = 10E-3f;
    uint64_t baseRates[NUM_PRE_NEURONS];
    convertRateToRandomNumberThreshold(&inputRate, &baseRates[0], 1);
    std::fill(&baseRates[1], &baseRates[NUM_PRE_NEURONS], baseRates[0]);

#ifndef CPU_ONLY
    // Copy base rates to GPU
    uint64_t *d_baseRates = NULL;
    CHECK_CUDA_ERRORS(cudaMalloc(&d_baseRates, sizeof(uint64_t) * NUM_PRE_NEURONS));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_baseRates, baseRates, sizeof(uint64_t) * NUM_PRE_NEURONS, cudaMemcpyHostToDevice));
    copyStateToDevice();
    ratesStim = d_baseRates;
#else
    ratesStim = baseRates;
#endif

    // Setup reverse connection indices for benchmark
    initbenchmark();

    auto  initEnd = chrono::steady_clock::now();
    printf("Init %ldms\n", chrono::duration_cast<chrono::milliseconds>(initEnd - initStart).count());

    auto simStart = chrono::steady_clock::now();
    // Loop through timesteps
    for(unsigned int t = 0; t < 5000; t++)
    {
        // Simulate
#ifndef CPU_ONLY
        stepTimeGPU();
#else
        stepTimeCPU();
#endif
  }
  auto simEnd = chrono::steady_clock::now();
  printf("Simulation %ldms\n", chrono::duration_cast<chrono::milliseconds>(simEnd - simStart).count());

  return 0;
}
