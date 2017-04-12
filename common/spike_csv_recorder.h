#pragma once

#include <fstream>

//----------------------------------------------------------------------------
// SpikeCSVRecorder
//----------------------------------------------------------------------------
class SpikeCSVRecorder
{
public:
    SpikeCSVRecorder(const char *filename,  unsigned int *spkCnt, unsigned int *spk)
    : m_Stream(filename), m_SpkCnt(spkCnt), m_Spk(spk)
    {
        m_Stream << "Time [ms], Neuron ID" << std::endl;
    }

    void record(double t)
    {
        for(unsigned int i = 0; i < m_SpkCnt[0]; i++)
        {
            m_Stream << t << "," << m_Spk[i] << std::endl;
        }
    }

private:

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_Stream;
    unsigned int *m_SpkCnt;
    unsigned int *m_Spk;
};

//----------------------------------------------------------------------------
// SpikeCSVRecorderDelay
//----------------------------------------------------------------------------
class SpikeCSVRecorderDelay
{
public:
    SpikeCSVRecorderDelay(const char *filename, unsigned int popSize, unsigned int &spkQueuePtr, unsigned int *spkCnt, unsigned int *spk)
    : m_Stream(filename), m_SpkQueuePtr(spkQueuePtr), m_SpkCnt(spkCnt), m_Spk(spk), m_PopSize(popSize)
    {
        m_Stream << "Time [ms], Neuron ID" << std::endl;
    }

    void record(double t)
    {
        unsigned int *currentSpk = getCurrentSpk();
        for(unsigned int i = 0; i < getCurrentSpkCnt(); i++)
        {
            m_Stream << t << "," << currentSpk[i] << std::endl;
        }
    }

private:
    unsigned int *getCurrentSpk() const
    {
        return &m_Spk[m_SpkQueuePtr * m_PopSize];
    }

    unsigned int getCurrentSpkCnt() const
    {
        return m_SpkCnt[m_SpkQueuePtr];
    }
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_Stream;
    unsigned int &m_SpkQueuePtr;
    unsigned int *m_SpkCnt;
    unsigned int *m_Spk;
    unsigned int m_PopSize;
};