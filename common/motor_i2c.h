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
        
        template<typename T, size_t N>
        void read(T (&data)[N])
        {
            return m_I2C.read(data);
        }
        
        //---MOVE ROBOT------------------------
        // move robot 1 forward 2 backward 0 stop
        void tank(uint8_t left_wheel, uint8_t right_wheel) 
        {  
            uint8_t buffer[2] = { left_wheel, right_wheel };
            
            // sending command to the arduino
            m_I2C.write(buffer);
        }
        
    private:
        I2CInterface m_I2C;
};
