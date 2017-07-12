#pragma once

#include <fstream>

//----------------------------------------------------------------------------
// AnalogueRecorder
//----------------------------------------------------------------------------
template<typename T>
class AnalogueCSVRecorder
{
public:
    AnalogueCSVRecorder(const char *filename,  T *variable, unsigned int popSize, const char *columnHeading)
    : m_Stream(filename), m_Variable(variable), m_PopSize(popSize)
    {
        m_Stream << "Time [ms], Neuron ID," << columnHeading << std::endl;
    }

    void record(double t)
    {
        for(unsigned int i = 0; i <  m_PopSize; i++)
        {
            m_Stream << t << "," << i << "," << m_Variable[i] << std::endl;
        }
    }

private:

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::ofstream m_Stream;
    T *m_Variable;
    unsigned int m_PopSize;
};
