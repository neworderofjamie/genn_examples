#include "deep_r.h"

// Standard C++ includes
#include <iostream>
#include <numeric>

// Standard C includes
#include <cassert>

// CUDA includes
#include <curand_kernel.h>

// GeNN userproject includes
//#include "timer.h"

// Batch-learning includes
#include "cuda_helpers.h"
#include "optimisers.h"

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{

__global__ void initDeepRKernel(uint32_t *d_Bitmask, const unsigned int *d_RowLength, const unsigned int *d_Ind, curandState *d_RNG,
                                unsigned int numRows, unsigned int maxRowLength, unsigned int bitmaskRowWords, unsigned long long seed)
{
    const unsigned int idPre = (blockIdx.x * 32) + threadIdx.x;
    
    if(idPre < numRows) {
        uint32_t *bitmaskRow = &d_Bitmask[bitmaskRowWords * idPre];
        for(unsigned i = 0; i < d_RowLength[i]; i++) {
            const unsigned int ind = d_Ind[i];
            
            bitmaskRow[ind / 32] |= (1 << (ind % 32));
        }
        
        // Initialise RNG
        curand_init(seed, idPre, 0, &d_RNG[idPre]);
    }
}

template<typename Operation>
__global__ void deepRFirstPassKernel(float *d_G, float *d_EFiltered, unsigned *d_NumDormantConnections,
                                     unsigned int *d_RowLength, unsigned int *d_Ind, uint32_t *d_Bitmask, 
                                     unsigned int numRows, unsigned int maxRowLength, unsigned int bitmaskRowWords,
                                     Operation operation)
{
    const unsigned int idPre = (blockIdx.x * 32) + threadIdx.x;
    
    // Use first thread in block to zero shared memory dormant counter
     __shared__ unsigned int shNumDormant;
    if(threadIdx.x == 0) {
        shNumDormant = 0;
    }
    __syncthreads();
    
    // If there's a row for this thread to process
    if(idPre < numRows) {
        uint32_t *bitmaskRow = &d_Bitmask[bitmaskRowWords * idPre];
        
        // Loop through synapses
        unsigned int numDormant = 0;
        unsigned int rowLength = d_RowLength[idPre];
        const unsigned int rowStartIdx = idPre * maxRowLength;
        for(unsigned int j = 0; j < rowLength; j++) {
            const unsigned int idx = rowStartIdx + j;
            
            // Cache parameter and its sign in register
            float gIn = d_G[idx];
            const auto oldSign = signbit(gIn);
            
            // If update changes parameter
            // **TODO** L1 regularizer
            if(operation.updateParameter(gIn, idx)) {
                // If sign hasn't changed, update weight in memory
                if(signbit(gIn) == oldSign) {
                    d_G[idx] = gIn;
                }
                // Otherwise, make connection dormant
                else {
                    // Calculate index of last synapse in row
                    const unsigned int rowLastIdx = rowStartIdx + rowLength - 1;
                    
                    // Clear bit in bitmask
                    const unsigned int ind = d_Ind[idx];
                    bitmaskRow[ind / 32] &= ~(1 << (ind % 32));
                    
                    // Overwrite this synapse with one at end of row
                    d_Ind[idx] = d_Ind[rowLastIdx];
                    d_G[idx] = d_G[rowLastIdx];
                    d_EFiltered[idx] = d_EFiltered[rowLastIdx];
                    
                    // Instruct operation to do the same for any of its parameters
                    operation.moveParams(rowLastIdx, idx);
                    
                    // Decrement j so replacement synapse gets processed
                    j--;
                    
                    // Decrement row length
                    rowLength--;
                    
                    // Increment row's dormant counter
                    numDormant++;
                }
            }
        }
        
        // Write back updated row length
        d_RowLength[idPre] = rowLength;
        
        // Update shared memory dormant synapse count
        if(numDormant > 0) {
            atomicAdd(&shNumDormant, numDormant);
        }
    }
    
    // Use first thread in block to atomic add shared memory counter to global total
    __syncthreads();
    if(threadIdx.x == 0 && shNumDormant > 0) {
        atomicAdd(d_NumDormantConnections, shNumDormant);
    }
}

template<typename Operation>
__global__ void deepRSecondPassKernel(float *d_G, float *d_EFiltered, unsigned int *d_RowLength, unsigned int *d_Ind, 
                                      unsigned int numRows, unsigned int numCols, unsigned int maxRowLength, unsigned int bitmaskRowWords,
                                      uint32_t *d_Bitmask, unsigned int *d_NumActivations, curandState *d_RNG,
                                      Operation operation)
{
    const unsigned int idPre = (blockIdx.x * 32) + threadIdx.x;
    
    // If there's a row for this thread to process
    if(idPre < numRows) {
        uint32_t *bitmaskRow = &d_Bitmask[bitmaskRowWords * idPre];
        const unsigned int rowStartIdx = idPre * maxRowLength;
        
        // Copy RNG state to register
        curandState rng = d_RNG[idPre];
        
        // Loop through activations we need to distribute across this row
        unsigned int numActivations = d_NumActivations[idPre];
        for(unsigned int i = 0; i < numActivations; i++) {
            while(true) {
                // Pick a random synapse to activate
                const unsigned int j = curand(&rng) % numCols;
                
                // Split this into bitmask words and bits
                const unsigned int jWord = j / 32;
                const unsigned int jBit = 1 << (j % 32);
                
                // If this bit isn't already set
                if((bitmaskRow[jWord] & jBit) == 0) {
                    // Set bit
                    bitmaskRow[jWord] |= jBit;
                    
                    // Calculate index of new synapse in sparse structure
                    const unsigned int idx = rowStartIdx + d_RowLength[idPre];
                    
                    // Set postsynaptic index and zero eligibility trace
                    d_Ind[idx] = j;
                    d_EFiltered[idx] = 0.0f;
                    
                    // Initialise weight
                    // **HACK** GPU treatment of -0 and +0 is likely to vary due to compiler options so use small numbers instead
                    d_G[idx] = (curand_uniform(&rng) < 0.2f) ? -0.000000001f : 0.0000000001f;
                    
                    // Instruct operation to initialise any of its parameters
                    operation.initSynapse(idx);
                    
                    // Increment row length
                    d_RowLength[idPre]++;
                    
                    // Go onto next synapse
                    break;
                }
            }
        }
        
        // Write update RNG state back to global memory
        d_RNG[idPre] = rng;
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// DeepR
//----------------------------------------------------------------------------
DeepR::DeepR(unsigned int numRows, unsigned int numCols, unsigned int maxRowLength,
             unsigned int *rowLength, unsigned int *d_rowLength, unsigned int *d_ind, 
             float *d_DeltaG, float *d_M, float *d_V, float *d_G, float *d_EFiltered, 
             float beta1, float beta2, float epsilon, unsigned int seed)
:   m_NumRows(numRows), m_NumCols(numCols), m_MaxRowLength(maxRowLength),
    m_BitmaskRowWords((m_NumCols  + 31) / 32), m_RowLength(rowLength), md_RowLength(d_rowLength), md_Ind(d_ind),
    md_DeltaG(d_DeltaG), md_M(d_M), md_V(d_V), md_G(d_G), md_EFiltered(d_EFiltered),
    m_Beta1(beta1), m_Beta2(beta2), m_Epsilon(epsilon), m_HostUpdateTime(0.0)
{
    // Allocate additional arrays to hold number of activation
    CHECK_CUDA_ERRORS(cudaMalloc(&md_NumActivations, m_NumRows * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaHostAlloc(&m_NumActivations, m_NumRows * sizeof(unsigned int), cudaHostAllocPortable));
    
    // Allocate dormant connection counter
    CHECK_CUDA_ERRORS(cudaMalloc(&md_NumDormantConnections, sizeof(unsigned int)));
    
    // Allocate RNG state
    CHECK_CUDA_ERRORS(cudaMalloc(&md_RNG, m_NumRows * sizeof(curandState)));
    
    // Allocate bitmask
    CHECK_CUDA_ERRORS(cudaMalloc(&md_Bitmask, m_BitmaskRowWords * m_NumRows * sizeof(uint32_t)));
    
    // Zero bitmask
    CHECK_CUDA_ERRORS(cudaMemset(md_Bitmask, m_BitmaskRowWords * m_NumRows * sizeof(uint32_t), 0));
    
    // If no seed is passed 
    unsigned long long deviceSeed;
    if(seed == 0) {
        std::random_device seedSource;
        
        // Initialize device seed using seed source
        uint32_t *deviceSeedWord = reinterpret_cast<uint32_t*>(&deviceSeed);
        for(int i = 0; i < (sizeof(unsigned long long) / sizeof(uint32_t)); i++) {
            deviceSeedWord[i] = seedSource();
        }
        
        // Generate random state for host RNG from seed source
        uint32_t seedData[std::mt19937::state_size];
        for(int i = 0; i < std::mt19937::state_size; i++) {
            seedData[i] = seedSource();
        }
        
        // Convert into seed sequence
        std::seed_seq seeds(std::begin(seedData), std::end(seedData));
        m_RNG.seed(seeds);
    }
    // Otherwise
    else {
        // Use seed to create seed sequence for host RNG
        // **NOTE** this is a terrible idea see http://www.pcg-random.org/posts/cpp-seeding-surprises.html
        std::seed_seq seeds{seed};
        m_RNG.seed(seeds);
        
        // Use seed directly for device RNG
        deviceSeed = seed;
    }

    // Calculate number of blocks required to process matrix
    const unsigned int numBlocks = (numRows + 31) / 32;
    
    // Launch kernel to initialize Deep-R structures
    const dim3 threads(32, 1);
    const dim3 grid(numBlocks, 1);
    initDeepRKernel<<<grid, threads>>>(md_Bitmask, md_RowLength, md_Ind, md_RNG,
                                       m_NumRows, m_MaxRowLength, m_BitmaskRowWords, deviceSeed);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}
//----------------------------------------------------------------------------
DeepR::~DeepR()
{
    cudaFree(md_NumActivations);
    cudaFreeHost(m_NumActivations);
    cudaFree(md_NumDormantConnections);
    cudaFree(md_Bitmask);
}
//----------------------------------------------------------------------------
void DeepR::update(unsigned int t, float alpha)
{
    // Time function
    //TimerAccumulate b(m_HostUpdateTime);

    // Calculate number of blocks required to process matrix
    const unsigned int numBlocks = (m_NumRows + 31) / 32;

    // Create optimizer
    // **TODO** template
    AdamOptimizer adam(md_DeltaG, md_M, md_V, t, alpha, m_Beta1, m_Beta2, m_Epsilon);

    // Zero device dormant count
    unsigned int numDormant = 0;
    CHECK_CUDA_ERRORS(cudaMemcpy(md_NumDormantConnections, &numDormant, sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Launch kernel to perform first Deep-R pass
    const dim3 threads(32, 1);
    const dim3 grid(numBlocks, 1);
    m_FirstPassKernelTimer.start();
    deepRFirstPassKernel<<<grid, threads>>>(md_G, md_EFiltered, md_NumDormantConnections, 
                                            md_RowLength, md_Ind, md_Bitmask,
                                            m_NumRows, m_MaxRowLength, m_BitmaskRowWords, adam);
    m_FirstPassKernelTimer.stop();
    
    // Copy device dormant count back to host
    CHECK_CUDA_ERRORS(cudaMemcpy(&numDormant, md_NumDormantConnections, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Copy row lengths back to host
    CHECK_CUDA_ERRORS(cudaMemcpy(m_RowLength, md_RowLength, m_NumRows * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Count number of synapses
    const size_t numSynapses = std::accumulate(&m_RowLength[0], &m_RowLength[m_NumRows], 0u);
    
    // From this, calculate how many padding synapses there are in data structure
    size_t numTotalPaddingSynapses = (m_MaxRowLength * m_NumRows) - numSynapses;

    // Loop through rows of synaptic matrix
    for(unsigned int i = 0; i < (m_NumRows - 1); i++) {
        const unsigned int numRowPaddingSynapses = m_MaxRowLength - m_RowLength[i];
        const double probability = (double)numRowPaddingSynapses / (double)numTotalPaddingSynapses;

        // Create distribution to sample number of activations
        std::binomial_distribution<unsigned int> numActivationDist(numDormant, probability);

        // Sample number of activations
        const unsigned int numActivations = std::min(numRowPaddingSynapses, numActivationDist(m_RNG));
        m_NumActivations[i] = numActivations;

        // Update counters
        numDormant -= numActivations;
        numTotalPaddingSynapses -= numRowPaddingSynapses;
    }

    // Put remainder of activations in last row
    assert(numDormant < (m_MaxRowLength - m_RowLength[m_NumRows - 1]));
    m_NumActivations[m_NumRows - 1] = numDormant;

    // Copy number of activations to device
    CHECK_CUDA_ERRORS(cudaMemcpy(md_NumActivations, m_NumActivations, m_NumRows * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // Launch kernel to perform second Deep-R pass
    m_SecondPassKernelTimer.start();
    deepRSecondPassKernel<<<grid, threads>>>(md_G, md_EFiltered, md_RowLength, md_Ind, 
                                             m_NumRows, m_NumCols, m_MaxRowLength, m_BitmaskRowWords,
                                             md_Bitmask, md_NumActivations, md_RNG, adam);
    m_SecondPassKernelTimer.stop();
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}
