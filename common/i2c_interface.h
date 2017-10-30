#pragma once

// Standard C++ includes
#include <iostream>
#include <vector>

// Standard C includes
#include <cstring>

// Posix includes
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>

// I2C includes
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

//----------------------------------------------------------------------------
// I2CInterface
//----------------------------------------------------------------------------
class I2CInterface
{
public:
    I2CInterface() : m_I2C(0)
    {
    }
    
    I2CInterface(const char *path, int slaveAddress) : m_I2C(0)
    {
        if(!setup(path, slaveAddress)) {
            throw std::runtime_error("Cannot open I2C interface");
        }
    }
    
    ~I2CInterface()
    {
        // Close I2C device
        if(m_I2C >= 0) {
            close(m_I2C);
        }
    }
    
    //---------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------
    bool setup(const char *path, int slaveAddress ) 
    {
        m_I2C = open(path, O_RDWR);
        if (m_I2C < 0) {
            std::cerr << "Error in setup:" << strerror(errno) << std::endl;
            // the error is usually permission error for which we can
            // temporarily use < $sudo chmod 666 /dev/i2c-1 >
            return false;
        }

        if (ioctl(m_I2C, I2C_SLAVE, slaveAddress) < 0) {
            std::cerr << "Cannot connect to the slave" << std::endl;
            return false;
        }
        else {
            std::cout << "I2C successfully initialized" << std::endl;
            return true;
        }
    }
    
    template<typename T, size_t N>
    void read(T (&data)[N])
    {
        if (::read(m_I2C, &data[0], sizeof(T) * N) < 0) {
            std::cerr << "Failed to read from i2c bus" << std::endl;
        }
    }
    
    // writes data
    template<typename T, size_t N>
    void write(const T (&data)[N]) 
    {  
        if (::write(m_I2C, &data[0], sizeof(T) * N) < 0) {
            std::cerr << "Failed to write to i2c bus" << std::endl;
        }
    }

private:
    //---------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------
    int m_I2C;                                      // i2c file
};