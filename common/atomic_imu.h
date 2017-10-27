#pragma once

// Standard C++ includes
#include <algorithm>
#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>

// Standard C includes
#include <cstdint>
#include <cstring>

// Posix includes
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>

//----------------------------------------------------------------------------
// AtomicIMU
//----------------------------------------------------------------------------
class AtomicIMU
{
public:
    // Enumerations
    enum Channel
    {
        ChannelAccelX,
        ChannelAccelY,
        ChannelAccelZ,
        ChannelPitch,
        ChannelRoll,
        ChannelYaw,
        ChannelMax,
    };
    
    AtomicIMU(const std::string &device = "/dev/ttyTHS2")
    {
        if(!connect(device)) {
            throw std::runtime_error("Cannot connect to Atomic IMU connected to '" + device + "'");
        }
    }
    
    ~AtomicIMU()
    {
        // Set should quit flag and wait on read thread to finish
        m_ShouldQuit = true;
        m_ReadThread.join();
        
        // Close UART device
        if(m_UART >= 0) {
            // Send space to return to idle mode
            sendByte(' ');
            
            // Close UART
            close(m_UART);
        }
    }
    
    //---------------------------------------------------------------------
    // Public API
    //---------------------------------------------------------------------
    bool connect(const std::string &device = "/dev/ttyTHS2")
    {
        // Open UART
        m_UART = open(device.c_str(), O_RDWR | O_NOCTTY);
        if (m_UART < 0) {
            std::cerr << "Error opening UART:" << strerror(errno) << std::endl;
            return false;
        }
        
        // Read current UART options
        termios uartOptions;
        if(tcgetattr(m_UART, &uartOptions) < 0) {
            std::cerr << "Error getting UART options:" << strerror(errno) << std::endl;
            return false;
        }
        
        // Set baud rate to 115200 (in both directions)
        if(cfsetispeed(&uartOptions, B115200) < 0) {
            std::cerr << "Error setting UART speed:" << strerror(errno) << std::endl;
            return false;
        }
        if(cfsetospeed(&uartOptions, B115200) < 0) {
            std::cerr << "Error getting UART speed:" << strerror(errno) << std::endl;
            return false;
        }

        // Put UART into some sort of legacy mode that emulates 
        // 'raw mode of the olf Version 7 terminal driver'
        // (whatever the hell that means)
        cfmakeraw(&uartOptions);
        
        // No timeout
        uartOptions.c_cc[VTIME] = 0;
        
        // Blocks until at least one character is available
        uartOptions.c_cc[VMIN] = 1;
        
        // Set UART options immediately
        if(tcsetattr(m_UART, TCSANOW, &uartOptions) < 0) {
            std::cerr << "Error setting UART options:" << strerror(errno) << std::endl;
            return false;
        }
        
        // Clear atomic stop flag and start thread
        m_ShouldQuit = false;
        m_ReadThread = std::thread(&AtomicIMU::readThread, this);
        return true;
    }
    
    void read(float (&data)[ChannelMax])
    {
        // Copy raw data into buffer, guarding with read mutex
        uint16_t rawData[ChannelMax];
        {
            std::lock_guard<std::mutex> guard(m_DataMutex);
            std::copy_n(&m_Data[0], ChannelMax, &rawData[0]);
        }
        
        // Transform raw data into floating point values
        std::transform(&rawData[0], &rawData[ChannelMax], &data[0], 
                       [](uint16_t raw)
                       {
                           // Swap endianess
                           // **NOTE** 'network order' is big-endian like Atomic IMU
                           const uint16_t host = ntohs(raw);
                           
                           // Convert to float and scale to -1.0, 1.0 range
                           return ((float)host / 511.5f) - 1.0f;
                       });
    }
    
private:
    //---------------------------------------------------------------------
    // Private methods
    //---------------------------------------------------------------------
    void readThread()
    {
        // Now thread is ready, send command to start binary streaming
        if(!sendByte('#')) {
            std::cerr << "Cannot send start command" << std::endl;
            return;
        }
        
        // Sync with start of first frame
        bool frameStarted = false;
        for(unsigned int i = 0; i < ((2 * sizeof(char)) + (ChannelMax * sizeof(uint16_t))); i++) {
            char frameStart;
            if(::read(m_UART, &frameStart, sizeof(char)) != sizeof(char)) {
                std::cerr << "Cannot read frame start from UART" << std::endl;
                return;
            }

            if(frameStart == 'A') {
                frameStarted = true;
                break;
            }
        }
        if(!frameStarted) { 
            std::cerr << "Frame start character not received" << std::endl;
            return;
        }
        
        // Read data until quit flag is set
        while(!m_ShouldQuit) {
            // Read count 
            if(!readBytes(reinterpret_cast<char*>(&m_Count), sizeof(uint16_t))) {
                std::cerr << "Cannot read count from UART" << std::endl;
                break;
            }

            // Read channel data using mutex
            {
                std::lock_guard<std::mutex> guard(m_DataMutex);
                if(!readBytes(reinterpret_cast<char*>(&m_Data[0]), sizeof(uint16_t) * ChannelMax)) {
                    std::cerr << "Cannot read channel data from UART" << std::endl;
                    break;
                }
            }
        
            // Read frame end
            char frameEnd;
            if(::read(m_UART, &frameEnd, sizeof(char)) != sizeof(char)) {
                std::cerr << "Cannot read frame end from UART" << std::endl;
                break;
            }
            if(frameEnd != 'Z') {
                std::cerr << "Frame incorrectly ended with '" << frameEnd << "' character" << std::endl;
                break;
            }
            
            // Read start of next frame
            char frameStart;
            if(::read(m_UART, &frameStart, sizeof(char)) != sizeof(char)) {
                std::cerr << "Cannot read frame start from UART" << std::endl;
                break;
            }

            if(frameStart != 'A') {
                std::cerr << "Frame incorrectly started with '" << frameStart << "' character" << std::endl;
                break;
            }
        }
    }
    
    bool readBytes(char *bytes, unsigned int remainingBytesToRead) 
    {
        while(remainingBytesToRead > 0) {
            // Attempt to read bytes into channelData
            int bytesRead = ::read(m_UART, bytes, remainingBytesToRead);
            if(bytesRead < 0) {
                std::cerr << "Cannot read data from UART" << std::endl;
                return false;
            }
            // Update counter and data
            remainingBytesToRead -= bytesRead;
            bytes += bytesRead;
        }
        return true;
    }
    
    bool sendByte(char command)
    {
        if(write(m_UART, &command, sizeof(char)) != sizeof(char)) {
            return false;
        }
        else {
            return true;
        }
    }

    //---------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------
    std::atomic<bool> m_ShouldQuit;
    std::thread m_ReadThread;
    std::mutex m_DataMutex;
    uint16_t m_Count;
    uint16_t m_Data[ChannelMax];
    int m_UART;
};