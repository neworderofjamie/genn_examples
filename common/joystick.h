#pragma once

// Standard C++ includes
#include <algorithm>
#include <bitset>
#include <iostream>
#include <limits>

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
       if(m_Joystick >= 0) {
           close(m_Joystick);
       }
    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    bool open(const char *device)
    {
        // Open joystick for non-blocking IO
        m_Joystick = ::open(device, O_RDONLY | O_NONBLOCK);
        if (m_Joystick < 0) {
            std::cerr << "Could not open joystick device '" << device << "' (" << strerror(errno) << ")" << std::endl;
            return false;
        }
        else {
            return true;
        }
    }

    bool read()
    {
        while(true) {
            // Attempt to read event from joystick device (non-blocking)
            js_event event;
            const ssize_t bytesRead = ::read(m_Joystick, &event, sizeof(js_event));
            
            // If there was an error
            if(bytesRead == -1) {
                // If there are no more events, return true
                if(errno == EAGAIN) {
                    return true;
                }
                // Otherwise return false
                else {
                    std::cerr << "Error: Could not read from joystick (" << strerror(errno) << ")" << std::endl;
                    return false;
                }
            }
            // Otherwise, if an event was read
            else if(bytesRead == sizeof(js_event)){
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
                    continue;
                }
            }
            else {
                std::cerr << "Unknown error" << std::endl;
                return false;
            }
        }
        
    }
    
    bool isButtonDown(uint8_t button) 
    {
        return m_ButtonState[button];
    }

    float getAxisState(uint8_t axis) 
    {
        return (float)m_AxisState[axis] / (float)std::numeric_limits<int16_t>::max();
    }
    
    template<typename Motor>
    void driveMotor(Motor &motor, float deadzone = 0.25f) 
    {
        const float joystickX = getAxisState(0);
        const float joystickY = getAxisState(1);
        if(joystickX < -deadzone) {
            motor.tank(1.0f, -1.0f);
        }
        else if(joystickX > deadzone) {
            motor.tank(-1.0f, 1.0f);
        }
        else if(joystickY < -deadzone) {
            motor.tank(1.0f, 1.0f);
        }
        else if(joystickY > deadzone) {
            motor.tank(-1.0f, -1.0f);
        }
        else {
            motor.tank(0.0f, 0.0f);
        }
    }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    int m_Joystick;
    
    int16_t m_AxisState[std::numeric_limits<uint8_t>::max()];
    std::bitset<std::numeric_limits<uint8_t>::max()> m_ButtonState;
};