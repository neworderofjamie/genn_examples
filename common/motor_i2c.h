#pragma once

// Standard C++ includes
#include <vector>

// Standard C includes
#include <cstdint>

// Common includes
#include "i2c_interface.h"

//----------------------------------------------------------------------------
// MotorI2C
//----------------------------------------------------------------------------
class MotorI2C
{
public:
    MotorI2C(const char *path = "/dev/i2c-1", int slaveAddress = 0x29) : m_I2C(path, slaveAddress)
    {
    }
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void tank(float left, float right) 
    {  
        // Convert standard (-1,1) values to bytes in order to send to I2C slave
        uint8_t buffer[2] = { floatToI2C(left), floatToI2C(left) };
        
        // Send buffer
        write(buffer);
    }

    template<typename T, size_t N>
    void read(T (&data)[N])
    {
        m_I2C.read(data);
    }
    
    template<typename T, size_t N>
    void write(const T (&data)[N]) 
    {
        m_I2C.write(data);
    }
    
    
private:
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    uint8_t floatToI2C(float speed) 
    {
        // Forward = 1
        if(speed > 0.0f) {
            return 1;
        }
        // Backwards = 2
        else if(speed < 0.0f) {
            return 2;
        }
        // Stop = 0
        else {
            return 0;
        }
    }
    //----------------------------------------------------------------------------
    // Private members
    //----------------------------------------------------------------------------
    I2CInterface m_I2C;
};
