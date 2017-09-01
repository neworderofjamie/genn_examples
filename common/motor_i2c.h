#pragma once

// Standard C++ includes
#include <vector>

// Standard C includes
#include <cstdint>

// Common includes
#include "i2c_transfer.h"

class MotorI2C
{
    public:
        //----READ SENSORS--------------------
        std::vector<uint8_t> read_smell() {                                     // returns the current reading for the smell sensor
            return i2c_transfer.read_data();
        }
        
        //---MOVE ROBOT------------------------
        void move_robot(uint8_t left_wheel, uint8_t right_wheel) {  // move robot 1 forward 2 backward 0 stop
            uint8_t buffer[2];
            buffer[0] = left_wheel;
            buffer[1] = right_wheel;
            // sending command to the arduino
            i2c_transfer.write_data(buffer, 2);
        }
        
        //---STOP ROBOT------------------------
        void stop_robot() {                                         // stops the robot - same as move_robot(0,0)
            move_robot(0, 0);
        }
        //-------------------------------------

    private:
        I2C_transfer i2c_transfer;
};
