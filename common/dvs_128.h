#pragma once

// Lib CAER includes
#include <libcaercpp/devices/dvs128.hpp>

//----------------------------------------------------------------------------
// DVS128
//----------------------------------------------------------------------------
class DVS128
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

    DVS128(Polarity polarity, uint16_t deviceID = 1)
        : m_DVS128Handle(deviceID, 0, 0, ""), m_Polarity(polarity), m_Width(0), m_Height(0)
    {
        // Let's take a look at the information we have on the device.
        auto info = m_DVS128Handle.infoGet();

        std::cout << info.deviceString << " - ID: " << info.deviceID << ", Master: " << info.deviceIsMaster;
        std::cout << ", DVS X: " << info.dvsSizeX << ", DVS Y: " << info.dvsSizeY << ", Logic: " << info.logicVersion << std::endl;

        // Cache width and height
        m_Width = (unsigned int)info.dvsSizeX;
        m_Height = (unsigned int)info.dvsSizeY;

        // Send the default configuration before using the device.
        // No configuration is sent automatically!
        m_DVS128Handle.sendDefaultConfig();

        // Tweak some biases, to increase bandwidth in this case.
        m_DVS128Handle.configSet(DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PR, 695);
        m_DVS128Handle.configSet(DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_FOLL, 867);
    }

    ~DVS128()
    {
        stop();
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void start()
    {
        m_DVS128Handle.dataStart();
    }

    void stop()
    {
        m_DVS128Handle.dataStop();
    }

    void readEvents(unsigned int &spikeCount, unsigned int *spikes)
    {
        // Zero spike count
        spikeCount = 0;

        // Get data from DVS
        auto packetContainer = m_DVS128Handle.dataGet();
        if (packetContainer == nullptr) {
            return;
        }

        // Loop through packets
        for (auto &packet : *packetContainer)
        {
            // If packet's empty, skip
            if (packet == nullptr) {
                continue;
            }
            // Otherwise if this is a polarity event
            else if (packet->getEventType() == POLARITY_EVENT) {
                // Cast to polarity packet
                auto polarityPacket = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                // Loop through events
                for(const auto &event : *polarityPacket)
                {
                    // If polarity is one we care about
                    if(m_Polarity == Polarity::Both
                        || (m_Polarity == Polarity::On && event.getPolarity())
                        || (m_Polarity == Polarity::Off && !event.getPolarity())) {

                        // Convert x and y coordinate to GeNN address and add to spike vector
                        const unsigned int gennAddress = (event.getX() + (event.getY() * m_Width));
                        spikes[spikeCount++] = gennAddress;
                    }
                }

                assert(spikeCount < (m_Width * m_Height));
            }
        }
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
    libcaer::devices::dvs128 m_DVS128Handle;
    const Polarity m_Polarity;
    unsigned int m_Width;
    unsigned int m_Height;
};