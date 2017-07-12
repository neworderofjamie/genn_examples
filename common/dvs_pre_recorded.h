#pragma once

// Standard C++ includes
#include <fstream>
#include <string>

// Standard C includes
#include <cstdlib>

//----------------------------------------------------------------------------
// DVSPreRecorded
//----------------------------------------------------------------------------
class DVSPreRecorded
{
public:
    //------------------------------------------------------------------------
    // Enumerations
    //------------------------------------------------------------------------
    enum class Polarity
    {
        On,
        Off,
        Both,
    };

    DVSPreRecorded(const char *spikeFilename, Polarity polarity, double dt, bool flipY = false, unsigned int width = 128, unsigned int height = 128)
        : m_SpikeStream(spikeFilename), m_Polarity(polarity), m_FrameDurationUs((unsigned int)(dt * 1000.0)),
          m_FlipY(flipY), m_Width(width), m_Height(height), m_FirstSpike(true), m_FrameStartTimestamp(0)
    {
        assert(m_SpikeStream.good());

        // Read header line
        std::getline(m_SpikeStream, m_NextLine);

        // Read first spike line
        std::getline(m_SpikeStream, m_NextLine);
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

        // Loop through spikes in frame
        std::string cell;
        do
        {
            // Create string stream from line
            std::stringstream lineStream(m_NextLine);

            // Read timestamp from
            std::getline(lineStream, cell, ',');
            const unsigned int timestamp = std::stoul(cell);

            // If this is the first spike, use it's timestamp as the start of the first frame
            if(m_FirstSpike) {
                m_FrameStartTimestamp = timestamp;
                m_FirstSpike = false;
            }
            // Otherwise if this spike doesn't occur in this frame - leave for processing during next frame
            else if(timestamp > (m_FrameStartTimestamp + m_FrameDurationUs)) {
                break;
            }

            // Read X coordinate
            std::getline(lineStream, cell, ',');
            const unsigned int x = std::stoul(cell);

            // Read Y coordinate
            std::getline(lineStream, cell, ',');
            const unsigned int y = m_FlipY ? (127 - std::stoul(cell)) : std::stoul(cell);

            // Calculate row-major spike address
            const unsigned int address = x + (y * m_Width);

            // If we don't care about polarity
            if(m_Polarity == Polarity::Both) {
                spikes[spikeCount++] = address;
            }
            else {
                // Read polarity
                std::getline(lineStream, cell, ',');
                const unsigned int polarity = std::stoul(cell);

                if((m_Polarity == Polarity::On && (polarity == 1))
                    || (m_Polarity == Polarity::Off && (polarity == 0))) {
                    spikes[spikeCount++] = address;
                }
            }

            // Read next spike into buffer
            std::getline(m_SpikeStream, m_NextLine);
        }
        while(m_SpikeStream.good());

        // Update frame start timestamp for next frame
        m_FrameStartTimestamp += m_FrameDurationUs;
    }

    unsigned int getWidth() const
    {
        return m_Width;
    }

    unsigned int getHeight() const
    {
        return m_Height;
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    std::ifstream m_SpikeStream;
    const Polarity m_Polarity;
    const unsigned int m_FrameDurationUs;
    const bool m_FlipY;
    const unsigned int m_Width;
    const unsigned int m_Height;
    std::string m_NextLine;
    bool m_FirstSpike;
    unsigned int m_FrameStartTimestamp;

};