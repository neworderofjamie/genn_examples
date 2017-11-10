#pragma once

// Standard C++ includes
#include <algorithm>
#include <atomic>
#include <bitset>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>

// Standard C includes
#include <cstdint>
#include <cstring>

// POSIX includes
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/joystick.h>

//----------------------------------------------------------------------------
// Joystick
//----------------------------------------------------------------------------
// Simple wrapper around Linux joystick to allow simple control of e.g. robots
class Joystick
{
public:
    Joystick(const char *device = "/dev/input/js0")
    {
        std::fill(std::begin(m_AxisState), std::end(m_AxisState), 0);
        if(!open(device)) {
            throw std::runtime_error("Cannot open joystick");
        }
    }

    ~Joystick()
    {
        // Set quit flag and join read thread
        m_ShouldQuit = true;
        m_ReadThread.join();
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool open(const char *device)
    {
        int joystick = ::open(device, O_RDONLY);
        if (joystick < 0) {
            std::cerr << "Could not open joystick device '" << device << "' (" << strerror(errno) << ")" << std::endl;
            return false;
        }

        // Clear atomic stop flag and start thread
        m_ShouldQuit = false;
        m_ReadThread = std::thread(&Joystick::readThread, this, joystick);
        return true;
    }

    bool isButtonDown(uint8_t button) {
        std::lock_guard<std::mutex> guard(m_StateMutex);
        return m_ButtonState[button];
    }

    float getAxisState(uint8_t axis) {
        std::lock_guard<std::mutex> guard(m_StateMutex);
        return (float)m_AxisState[axis] / (float)std::numeric_limits<int16_t>::max();
    }

private:
    //------------------------------------------------------------------------
    // Private methods
    //------------------------------------------------------------------------
    void readThread(int joystick)
    {
        js_event event;
        while(!m_ShouldQuit) {
            // Read from joystick device (blocking)
            if(::read(joystick, &event, sizeof(js_event)) != sizeof(js_event)) {
                std::cerr << "Error: Could not read from joystick (" << strerror(errno) << ")" << std::endl;
                break;
            }

            {
                // Lock state mutex
                std::lock_guard<std::mutex> guard(m_StateMutex);

                // If event is axis, copy value into axis
                // **NOTE** initial state is specified by ORing these
                // types with JS_EVENT_INIT so this test handles initial state too
                if((event.type & JS_EVENT_AXIS) != 0) {
                    m_AxisState[event.number] = event.value;
                }
                // Otherwise, if a button state has changed
                else if((event.type & JS_EVENT_BUTTON) != 0) {
                    m_ButtonState.set(event.number, event.value);
                }
                else {
                    std::cerr << "Unknown event type " << (unsigned int)event.type << std::endl;
                }
            }

        }

        // Close joystick
        close(joystick);
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::atomic<bool> m_ShouldQuit;
    std::thread m_ReadThread;

    std::mutex m_StateMutex;
    int16_t m_AxisState[std::numeric_limits<uint8_t>::max()];
    std::bitset<std::numeric_limits<uint8_t>::max()> m_ButtonState;
};