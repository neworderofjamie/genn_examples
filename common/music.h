#pragma once

// Music includes
#include <music.hh>

//----------------------------------------------------------------------------
// MUSICSpikeOut
//----------------------------------------------------------------------------
class MUSICSpikeOut
{
public:
    MUSICSpikeOut(const char *portName, unsigned int popSize,
                  const unsigned int *spkCnt, const unsigned int *spk,
                  MUSIC::Setup *setup)
    :   m_Indices(0, popSize), m_SpkCnt(spkCnt), m_Spk(spk)
    {
        // Public port
        m_Port = setup->publishEventOutput(portName);

        // Map port
        m_Port->map(&m_Indices, MUSIC::Index::GLOBAL);
        std::cout << "Mapped out" << std::endl;

    }

    virtual ~MUSICSpikeOut()
    {
        delete m_Port;
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void record(double t)
    {
        for(unsigned int i = 0; i < m_SpkCnt[0]; i++) {
            m_Port->insertEvent(t / 1000.0, MUSIC::GlobalIndex(m_Spk[i]));
        }
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    MUSIC::LinearIndex m_Indices;
    MUSIC::EventOutputPort *m_Port;
    const unsigned int *m_SpkCnt;
    const unsigned int *m_Spk;
};

//----------------------------------------------------------------------------
// MUSICSpikeIn
//----------------------------------------------------------------------------
/*class MUSICSpikeIn
{
public:
    MUSICSpikeIn(const char *portName, unsigned int popSize,
                 const unsigned int &spkQueuePtr, const unsigned int *spkCnt, const unsigned int *spk,
                 MUSIC::Setup &setup)
    :   m_SpkQueuePtr(spkQueuePtr), m_SpkCnt(spkCnt), m_Spk(spk), m_PopSize(popSize)
    {
        // Public port
        m_Port = setup.publishEventOutput(portName);

        // Map port
        MUSIC::LinearIndex indices(0, popSize);
        m_Port->map(&indices, MUSIC::Index::GLOBAL);

    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void record(double t)
    {
        const unsigned int *currentSpk = &m_Spk[m_SpkQueuePtr * m_PopSize];
        for(unsigned int i = 0; i < m_SpkCnt[m_SpkQueuePtr]; i++) {
            m_Port->insertEvent(t / 1000.0, MUSIC::GlobalIndex(currentSpk[i]));
        }
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    MUSIC::EventOutputPort *m_Port;
    const unsigned int &m_SpkQueuePtr;
    const unsigned int *m_SpkCnt;
    const unsigned int *m_Spk;
    const unsigned int m_PopSize;
};*/