#pragma once

// Auto-generated model code
#include "pattern_recognition_1_1_CODE/definitionsInternal.h"

//----------------------------------------------------------------------------
// CUDATimer
//----------------------------------------------------------------------------
class CUDATimer
{
public:
    CUDATimer() : m_TotalTime(0.0f)
    {
        CHECK_CUDA_ERRORS(cudaEventCreate(&m_StartEvent));
        CHECK_CUDA_ERRORS(cudaEventCreate(&m_StopEvent));
    }
    
    ~CUDATimer()
    {
        cudaEventDestroy(m_StartEvent);
        cudaEventDestroy(m_StopEvent);
    }
    
    void start()
    {
        CHECK_CUDA_ERRORS(cudaEventRecord(m_StartEvent));
    }
    
    void stop()
    {
        CHECK_CUDA_ERRORS(cudaEventRecord(m_StopEvent));
    }
    
    void synchronize()
    {
        CHECK_CUDA_ERRORS(cudaEventSynchronize(m_StopEvent));
    }
    
    void update()
    {
        float tmp;
        CHECK_CUDA_ERRORS(cudaEventElapsedTime(&tmp, m_StartEvent, m_StopEvent));
        m_TotalTime += (tmp / 1000.0f);
    }
    
    float getTotalTime() const{ return m_TotalTime; }
    
private:
    cudaEvent_t m_StartEvent;
    cudaEvent_t m_StopEvent;
    
    float m_TotalTime;
};