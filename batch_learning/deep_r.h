#pragma once

// Standard C++ includes
#include <random>

// eProp includes
#include "cuda_timer.h"

// Forward declarations
typedef struct curandStateXORWOW curandState;

//----------------------------------------------------------------------------
// BatchLearning::Deep-R
//----------------------------------------------------------------------------
namespace BatchLearning
{
class DeepR
{
public:
    DeepR(unsigned int numRows, unsigned int numCols, unsigned int maxRowLength,
          unsigned int *rowLength, unsigned int *d_rowLength, unsigned int *d_ind,
          float *d_DeltaG, float *d_M, float *d_V, float *d_G, float *d_EFiltered,
          float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1E-8, unsigned int seed = 0);
    ~DeepR();

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void update(unsigned int t, float alpha = 0.001);

    void updateTimers()
    {
        m_FirstPassKernelTimer.update();
        m_SecondPassKernelTimer.update();
    }

    float getFirstPassKernelTime() const { return m_FirstPassKernelTimer.getTotalTime(); }
    float getSecondPassKernelTime() const { return m_SecondPassKernelTimer.getTotalTime(); }
    double getHostUpdateTime() const { return m_HostUpdateTime; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    // Dimensions of matrix
    const unsigned int m_NumRows;
    const unsigned int m_NumCols;
    const unsigned int m_MaxRowLength;
    const unsigned int m_BitmaskRowWords;

    // GeNN-allocated row lengths for sparse connection
    unsigned int *m_RowLength;
    unsigned int *md_RowLength;

    // GeNN-allocated indices for sparse connection
    unsigned int *md_Ind;

    // GeNN-allocated synapse variables
    float *md_DeltaG;
    float *md_M;
    float *md_V;
    float *md_G;
    float *md_EFiltered;

    // Additional bitmask data structure used for Deep-R update
    uint32_t *md_Bitmask;

    // Device and host arrays to hold number of 
    // new activations to make for each row
    unsigned int *md_NumActivations;
    unsigned int *m_NumActivations;

    // Counter used to track number of dormant connections
    unsigned int *md_NumDormantConnections;

    // State for device RNG used for distributing re-activated synapses
    curandState *md_RNG;

    // Adam optimizer parameters
    // **THINK** this is probably not a good approach
    const float m_Beta1;
    const float m_Beta2;
    const float m_Epsilon;

    // RNG for distributing reactivations
    std::mt19937 m_RNG;

    // CUDA timers for two kernels
    CUDATimer m_FirstPassKernelTimer;
    CUDATimer m_SecondPassKernelTimer;
    double m_HostUpdateTime;
};
}   // namespace BatchLearning