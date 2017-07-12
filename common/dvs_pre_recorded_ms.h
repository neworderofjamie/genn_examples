#pragma once

// Standard C++ includes
#include <fstream>
#include <string>
#include <vector>

// Standard C includes
#include <cstdlib>

//----------------------------------------------------------------------------
// DVSPreRecordedMs
//----------------------------------------------------------------------------
class DVSPreRecordedMs
{
public:
    DVSPreRecordedMs(const char *spikeFilename)
        : m_SpikeStream(spikeFilename), m_NextInputTimestep(0), m_Timestep(0), m_MoreSpikes(false)
    {
        assert(m_SpikeStream.good());

        // Read next input
        m_MoreSpikes = readNext();
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void start()
    {
    }

    void stop()
    {
    }

    void readEvents(unsigned int &spikeCount, unsigned int *spikes)
    {
        // Zero spike count
        spikeCount = 0;

        // If we should supply input this timestep
        if(m_MoreSpikes && m_NextInputTimestep == m_Timestep) {
            // Copy into spike source
            spikeCount = m_NextInputAddresses.size();
            std::copy(m_NextInputAddresses.cbegin(), m_NextInputAddresses.cend(), spikes);

            // Read NEXT input
            m_MoreSpikes = readNext();
        }

        // Update internal timestep counter
        m_Timestep++;
    }

    unsigned int getWidth() const
    {
        return 128;
    }

    unsigned int getHeight() const
    {
        return 128;
    }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    bool readNext()
    {
         // Read lines into string
        std::string line;
        std::getline(m_SpikeStream, line);

        if(line.empty()) {
            return false;
        }

        // Create string stream from line
        std::stringstream lineStream(line);

        // Read time from start of line
        std::string nextTimeString;
        std::getline(lineStream, nextTimeString, ';');
        m_NextInputTimestep = (unsigned int)std::stoul(nextTimeString);

        // Clear existing addresses
        m_NextInputAddresses.clear();

        while(lineStream.good()) {
            // Read input spike index
            std::string inputIndexString;
            std::getline(lineStream, inputIndexString, ',');
            m_NextInputAddresses.push_back(std::atoi(inputIndexString.c_str()));
        }

        return true;
    }

    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::ifstream m_SpikeStream;

    std::vector<unsigned int> m_NextInputAddresses;
    unsigned int m_NextInputTimestep;
    unsigned int m_Timestep;
    bool m_MoreSpikes;

};