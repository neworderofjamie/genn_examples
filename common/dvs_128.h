#pragma once

// Lib CAER includes
#include <libcaercpp/devices/davis.hpp>
#include <libcaercpp/devices/dvs128.hpp>
#include <libcaercpp/devices/dvxplorer.hpp>

//----------------------------------------------------------------------------
// DVS::Polarity
//----------------------------------------------------------------------------
namespace DVS
{
enum class Polarity
{
    On,
    Off,
    Both,
};

//----------------------------------------------------------------------------
// DVS::Base
//----------------------------------------------------------------------------
template<typename Device>
class Base
{
public:
    Base(Polarity polarity, uint16_t deviceID = 1)
        : m_DVSHandle(deviceID), m_Polarity(polarity), m_Width(0), m_Height(0)
    {
        // Let's take a look at the information we have on the device.
        std::cout << m_DVSHandle.toString() << std::endl;

        // Cache width and height
        auto info = m_DVSHandle.infoGet();
        m_Width = (unsigned int)info.dvsSizeX;
        m_Height = (unsigned int)info.dvsSizeY;

        // Send the default configuration before using the device.
        // No configuration is sent automatically!
        m_DVSHandle.sendDefaultConfig();

        
    
        // Tweak some biases, to increase bandwidth in this case.
        //m_DVSHandle.configSet(DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PR, 695);
        //m_DVSHandle.configSet(DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_FOLL, 867);
    }

    ~Base()
    {
        stop();
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void start()
    {
        m_DVSHandle.dataStart(nullptr, nullptr, nullptr, nullptr, nullptr);
        
        // Let's turn on blocking data-get mode to avoid wasting resources.
        m_DVSHandle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);
    }

    void stop()
    {
        m_DVSHandle.dataStop();
    }

    void readEvents(unsigned int &spikeCount, unsigned int *spikes)
    {
        // Zero spike count
        spikeCount = 0;

        // Get data from DVS
        auto packetContainer = m_DVSHandle.dataGet();
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
    Device m_DVSHandle;
    const Polarity m_Polarity;
    unsigned int m_Width;
    unsigned int m_Height;
};

using DVS128 = Base<libcaer::devices::dvs128>;
using DVXplorer = Base<libcaer::devices::dvXplorer>;
using Davis = Base<libcaer::devices::davis>;
}
