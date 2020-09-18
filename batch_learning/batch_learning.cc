#include "batch_learning.h"

// Standard C++ includes
#include <string>
#include <stdexcept>

// CUDA includes
#include <curand_kernel.h>

// Batch-learning includes
#include "optimisers.h"

//----------------------------------------------------------------------------
// Macros
// ------------------------------------------------------------------------
// Helper macro for error-checking CUDA calls
#define CHECK_CUDA_ERRORS(call) {\
    cudaError_t error = call;\
    if (error != cudaSuccess) {\
        throw std::runtime_error(__FILE__": " + std::to_string(__LINE__) + ": cuda error " + std::to_string(error) + ": " + cudaGetErrorString(error));\
    }\
}

//----------------------------------------------------------------------------
// Anonymous namespace
//----------------------------------------------------------------------------
namespace
{
// How large are (square) tiles used to calculate CUDA transpose
constexpr size_t TILE_DIM = 32;

// How 'high' are thread blocks
constexpr size_t BLOCK_HEIGHT = 8;

class NOP
{
public:
    __forceinline__ __device__ bool updateParameter(float &, unsigned int)
    {
        return false;
    }
    
    __forceinline__ __device__ void moveParams(unsigned int, unsigned int)
    {
    }
};

// Optimised CUDA transpose kernel based on https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template<typename Operation>
__global__ void transposeKernel(float *d_in, float *d_out, 
                                unsigned int numInRows, unsigned int numInCols,
                                Operation operation)
{
    // **NOTE** adding extra column prevents conflicts between 32 shared memory banks
    __shared__ float shTile[TILE_DIM][TILE_DIM + 1];

    {
        // Calculate coordinate of thread in input matrix
        const unsigned int x = (blockIdx.x * TILE_DIM) + threadIdx.x;
        const unsigned int y = (blockIdx.y * TILE_DIM) + threadIdx.y;
        
        // If thread isn't off the 'right' edge of the input matrix
        if(x < numInCols) {
            // Loop through input rows 
            for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_HEIGHT) {
                // If thread isn't off the 'bottom' edge of the input matrix
                if((y + j) < numInRows) {
                    // Read forward weight from global memory
                    const unsigned int idxIn = ((y + j) * numInCols) + x;
                    float gIn = d_in[idxIn];
                    
                    // Update parameter - if it's changed update global memory
                    if(operation.updateParameter(gIn, idxIn)) {
                        d_in[idxIn] = gIn;
                    }
                    
                    // Write forward weight to share memory
                    shTile[threadIdx.y + j][threadIdx.x] = gIn;
                }
            }
        }
    }
    
    __syncthreads();

    {
        // Calculate (transposed) coordinate of thread in output matrix
        const unsigned int x = (blockIdx.y * TILE_DIM) + threadIdx.x;
        const unsigned int y = (blockIdx.x * TILE_DIM) + threadIdx.y;
        
        // If thread isn't off the 'right' edge of the output matrix
        if(x < numInRows) {
            // Loop through output rows
            for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_HEIGHT) {
                // If thread isn't off the 'bottom' edge of the output matrix
                if((y + j) < numInCols) {
                    d_out[((y + j) * numInRows) + x] = shTile[threadIdx.x][threadIdx.y + j];
                }
            }
        }
    }
}

template<typename Operation>
__global__ void updateKernel(float *d_G, unsigned int numSynapses, Operation operation)
{
    const unsigned int idx = (blockIdx.x * 32) + threadIdx.x;

    if(idx < numSynapses) {
        // Update parameter - if it's changed update global memory
        float gIn = d_G[idx];
        if(operation.updateParameter(gIn, idx)) {
            d_G[idx] = gIn;
        }
    }
}
}   // Anonymous namespace

//----------------------------------------------------------------------------
// BatchLearning
//----------------------------------------------------------------------------
namespace BatchLearning
{
void transposeCUDA(float *d_in, float *d_out, unsigned int numInRows, unsigned int numInCols)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;
    
    NOP nop;
    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    transposeKernel<<<grid, threads>>>(d_in, d_out, numInRows, numInCols, nop);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void fixedRateLearningCUDA(float *d_DeltaG, float *d_G, unsigned int numRows, unsigned int numCols, float learningRate)
{
    const unsigned int numSynapses = numRows * numCols;
    const unsigned int numBlocks = (numSynapses + 31) / 32;
    
    FixedLearningRate fixedLearningRate(d_DeltaG, learningRate);
    
    const dim3 threads(32, 1);
    const dim3 grid(numBlocks, 1);
    updateKernel<<<grid, threads>>>(d_G, numSynapses, fixedLearningRate);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void fixedRateLearningTransposeCUDA(float *d_DeltaGIn, float *d_GIn, float *d_GOut, unsigned int numInRows, unsigned int numInCols, float learningRate)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;
    
    FixedLearningRate fixedLearningRate(d_DeltaGIn, learningRate);
    
    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    transposeKernel<<<grid, threads>>>(d_GIn, d_GOut, numInRows, numInCols, fixedLearningRate);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void adamOptimizerCUDA(float *d_DeltaG, float *d_M, float *d_V, float *d_G, 
                       unsigned int numRows, unsigned int numCols, unsigned int t, 
                       float alpha, float beta1, float beta2, float epsilon)
{
    const unsigned int numSynapses = numRows * numCols;
    const unsigned int numBlocks = (numSynapses + 31) / 32;
    
    AdamOptimizer adam(d_DeltaG, d_M, d_V, t, alpha, beta1, beta2, epsilon);
    
    const dim3 threads(32, 1);
    const dim3 grid(numBlocks, 1);
    updateKernel<<<grid, threads>>>(d_G, numSynapses, adam);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void adamOptimizerTransposeCUDA(float *d_DeltaGIn, float *d_MIn, float *d_VIn, float *d_GIn, 
                                float *d_GOut, unsigned int numInRows, unsigned int numInCols, 
                                unsigned int t, float alpha, float beta1, 
                                float beta2, float epsilon)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;
    
    AdamOptimizer adam(d_DeltaGIn, d_MIn, d_VIn, t, alpha, beta1, beta2, epsilon);
    
    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    transposeKernel<<<grid, threads>>>(d_GIn, d_GOut, numInRows, numInCols, adam);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void rMaxPropCUDA(float *d_M, float *d_Epsilon, float *d_G,
                  unsigned int numRows, unsigned int numCols,
                  float updateTime, float tauRMS, float r0, float epsilon, float wMin, float wMax)
{
    const unsigned int numSynapses = numRows * numCols;
    const unsigned int numBlocks = (numSynapses + 31) / 32;

    RMaxProp rMaxProp(d_M, d_Epsilon, updateTime, tauRMS, r0, epsilon, wMin, wMax);

    const dim3 threads(32, 1);
    const dim3 grid(numBlocks, 1);
    updateKernel<<<grid, threads>>>(d_G, numSynapses, rMaxProp);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

void rMaxPropTransposeCUDA(float *d_MIn, float *d_EpsilonIn, float *d_GIn,
                           float *d_GOut, unsigned int numInRows, unsigned int numInCols,
                           float updateTime, float tauRMS, float r0, float epsilon, float wMin, float wMax)
{
    // Calculate number of blocks required to process matrix
    const unsigned int numBlockX = (numInCols + TILE_DIM - 1) / TILE_DIM;
    const unsigned int numBlockY = (numInRows + TILE_DIM - 1) / TILE_DIM;

    RMaxProp rMaxProp(d_MIn, d_EpsilonIn, updateTime, tauRMS, r0, epsilon, wMin, wMax);

    const dim3 threads(32, BLOCK_HEIGHT);
    const dim3 grid(numBlockX, numBlockY);
    transposeKernel<<<grid, threads>>>(d_GIn, d_GOut, numInRows, numInCols, rMaxProp);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
}

}   // namespace BatchLearning
