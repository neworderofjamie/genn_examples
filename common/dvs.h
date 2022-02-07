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
    ON,
    OFF,
    BOTH,
};

template<Polarity polarity = Polarity::BOTH>
struct PolarityFilter
{
    static constexpr bool shouldAllow(const libcaer::events::PolarityEvent &event)
    {
        return (polarity == Polarity::BOTH
                || (polarity == Polarity::ON && event.getPolarity())
                || (polarity == Polarity::OFF && !event.getPolarity()));
    }
};

template<uint16_t minX, uint16_t maxX, uint16_t minY, uint16_t maxY>
struct CropFilter
{
    static constexpr bool shouldAllow(const libcaer::events::PolarityEvent &event)
    {
        return (event.getX() > minX && event.getX() < maxX 
                && event.getY() > minY && event.getY() < maxY);
    }
};


template<typename FilterA, typename FilterB>
struct CombineFilter
{
    static constexpr bool shouldAllow(const libcaer::events::PolarityEvent &event)
    {
        return (FilterA::shouldAllow(event) && FilterB::shouldAllow(event));
    }
};

struct NoFilter
{
    static constexpr bool shouldAllow(const libcaer::events::PolarityEvent&)
    {
        return true;
    }
};

template<uint32_t FixedPointScale>
struct Scale
{
    static constexpr uint32_t transform(uint32_t x)
    {
        return (x * FixedPointScale) >> 15;
    }
};

template<uint32_t Offset>
struct Subtract
{
    static constexpr uint32_t transform(uint32_t x)
    {
        return (x - Offset);
    }
};

struct NoTransform
{
    static constexpr uint32_t transform(uint32_t x)
    {
        return x;
    }
};

template<typename TransformA, typename TransformB>
struct CombineTransform
{
    static constexpr uint32_t transform(uint32_t x)
    {
        return TransformB::transform(TransformA::transform(x));
    }
};

//----------------------------------------------------------------------------
// DVS::Base
//----------------------------------------------------------------------------
template<typename Device>
class Base
{
public:
    Base(uint16_t deviceID = 1)
        : m_DVSHandle(deviceID), m_Width(0), m_Height(0)
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
    }

    void stop()
    {
        m_DVSHandle.dataStop();
    }
    
    void configSet(int8_t modAddr, uint8_t paramAddr, uint32_t param)
    {
        m_DVSHandle.configSet(modAddr, paramAddr, param);
    }

    template<typename Filter = NoFilter, typename TransformX = NoTransform, typename TransformY = NoTransform, unsigned int outputSize>
    void readEvents(uint32_t *spikeVector)
    {
        // Get data from DVS
        auto packetContainer = m_DVSHandle.dataGet();
        if (packetContainer == nullptr) {
            return;
        }

        // Loop through packets
        for(auto &packet : *packetContainer) {
            // If packet's empty, skip
            if (packet == nullptr) {
                continue;
            }
            // Otherwise if this is a polarity event
            else if (packet->getEventType() == POLARITY_EVENT) {
                // Cast to polarity packet
                auto polarityPacket = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                // Loop through events
                for(const auto &event : *polarityPacket) {
                    // If event isn't filtered
                    if(Filter::shouldAllow(event)) {
                        // Transform event
                        const uint32_t transformX = TransformX::transform(event.getX());
                        const uint32_t transformY = TransformY::transform(event.getY());
                        
                        // Convert transformed X and Y into GeNN address
                        const unsigned int gennAddress = (transformX + (transformY * outputSize));
                        
                        // Set spike bit
                        spikeVector[gennAddress / 32] |= (1 << (gennAddress % 32));
                    }
                }
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
    unsigned int m_Width;
    unsigned int m_Height;
};

using DVS128 = Base<libcaer::devices::dvs128>;
using DVXplorer = Base<libcaer::devices::dvXplorer>;
using Davis = Base<libcaer::devices::davis>;
}
