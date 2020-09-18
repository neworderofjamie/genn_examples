#pragma once

// CUDA includes
#include <curand_kernel.h>

//----------------------------------------------------------------------------
// FixedLearningRate
//----------------------------------------------------------------------------
//! Simple 'operation' to use with transpose and update kernels to perform learning with a fixed rate
class FixedLearningRate
{
public:
    FixedLearningRate(float *gradients, float learningRate)
    :   m_Gradients(gradients), m_LearningRate(learningRate)
    {
    }
    
    __forceinline__ __device__ bool updateParameter(float &param, unsigned int idx)
    {
        // Subtract gradient to parameter, scaled by learning rate
        param -= (m_Gradients[idx] * m_LearningRate);
        
        // Zero gradient
        m_Gradients[idx] = 0.0f;
        return true;
    }
    
    __forceinline__ __device__ void moveParams(unsigned int srcIdx, unsigned int dstIdx)
    {
        m_Gradients[dstIdx] = m_Gradients[srcIdx];
    }
    
    __forceinline__ __device__ void initSynapse(unsigned int idx)
    {
        m_Gradients[idx] = 0.0f;
    }
    
private:
    float *m_Gradients;
    const float m_LearningRate;
};

//----------------------------------------------------------------------------
// AdamOptimizer
//----------------------------------------------------------------------------
//! Simple 'operation' to apply Adam optimizer to parameter in transpose and update kernels
class AdamOptimizer
{
public:
    AdamOptimizer(float *gradients, float *m, float *v, unsigned int t, float alpha = 0.001, 
                  float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1E-8)
    :   m_Gradients(gradients), m_M(m), m_V(v), m_Alpha(alpha), 
        m_Beta1(beta1), m_Beta2(beta2), m_Epsilon(epsilon), 
        m_FirstMomentScale(1.0f / (1.0f - pow(m_Beta1, t + 1))),
        m_SecondMomentScale(1.0f / (1.0f - pow(m_Beta2, t + 1)))
    {
    }
    
    __forceinline__ __device__ bool updateParameter(float &param, unsigned int idx)
    {
        // Get gradients
        const float gradient = m_Gradients[idx];
        
        // Update biased first moment estimate
        const float mT = (m_Beta1 * m_M[idx]) + ((1.0f - m_Beta1) * gradient);
        
        // Update biased second moment estimate
        const float vT = (m_Beta2 * m_V[idx]) + ((1.0f - m_Beta2) * gradient * gradient);
        
        // Add gradient to parameter, scaled by learning rate
        param -= (m_Alpha * mT * m_FirstMomentScale) / (sqrt(vT * m_SecondMomentScale) + m_Epsilon);
        
        // Write moments back to memory
        m_M[idx] = mT;
        m_V[idx] = vT;
        
        // Zero gradient
        m_Gradients[idx] = 0.0f;
        return true;
    }
    
    __forceinline__ __device__ void moveParams(unsigned int srcIdx, unsigned int dstIdx)
    {
        m_Gradients[dstIdx] = m_Gradients[srcIdx];
        m_M[dstIdx] = m_M[srcIdx];
        m_V[dstIdx] = m_M[dstIdx];
    }
    
    __forceinline__ __device__ void initSynapse(unsigned int idx)
    {
        m_Gradients[idx] = 0.0f;
        m_M[idx] = 0.0f;
        m_V[idx] = 0.0f;
    }
    
private:
    float *m_Gradients;
    float *m_M;
    float *m_V;
    const float m_Alpha;
    const float m_Beta1;
    const float m_Beta2;
    const float m_Epsilon;
    const float m_FirstMomentScale;
    const float m_SecondMomentScale;
};
