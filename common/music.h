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
        // Publish port
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
    void transmit(double t)
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
    const unsigned int * const m_SpkCnt;
    const unsigned int * const m_Spk;
};

//----------------------------------------------------------------------------
// MUSICSpikeIn
//----------------------------------------------------------------------------
class MUSICSpikeIn : public MUSIC::EventHandlerGlobalIndex
{
public:
    MUSICSpikeIn(const char *portName, unsigned int popSize, double dt,
                 unsigned int * const spkCnt, unsigned int * const spk,
                 MUSIC::Setup *setup)
    :   m_Indices(0, popSize), m_SpkCnt(spkCnt), m_Spk(spk)
    {
        // Publish port
        m_Port = setup->publishEventInput(portName);

        // Map port
        m_Port->map(&m_Indices, this, dt / 1000.0, 1);
    }

    virtual ~MUSICSpikeIn()
    {
        delete m_Port;
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void tick()
    {
        m_SpkCnt[0] = 0;
    }

    void operator () (double t, MUSIC::GlobalIndex id)
    {
        // Add incoming spike to buffer
        m_Spk[m_SpkCnt[0]++]= id;
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    MUSIC::LinearIndex m_Indices;
    MUSIC::EventInputPort *m_Port;
    unsigned int * const m_SpkCnt;
    unsigned int * const m_Spk;
};