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
#include <cmath>
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
        ChannelAccelMax,
        ChannelGyroPitch = ChannelAccelMax,
        ChannelGyroRoll,
        ChannelGyroYaw,
        ChannelGyroMax,
        ChannelMax = ChannelGyroMax,
    };
    
    AtomicIMU(const std::string &device = "/dev/ttyTHS2") : m_Orientation{1.0f, 0.0f, 0.0f, 0.0f}
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
    
    void readData(float (&data)[ChannelMax])
    {
        // Copy data into buffer, guarding with mutex
        {
            std::lock_guard<std::mutex> guard(m_DataMutex);
            std::copy_n(&m_Data[0], ChannelMax, &data[0]);
        }
    }
    
    void readEuler(float (&euler)[3])
    {
        // Read orientation
        float orientation[4];
        readOrientation(orientation);
        
        // Calculate euler angles
        euler[0] = atan2((2.0f * orientation[1] * orientation[2]) - (2.0f * orientation[0] * orientation[3]),
                         (2.0f * orientation[0] * orientation[0]) + (2.0f * orientation[1] * orientation[1]) - 1.0f);
        euler[1] = -asin((2.0f * orientation[1] * orientation[3]) + (2.0f * orientation[0] * orientation[2]));
        euler[2] = atan2((2.0f * orientation[2] * orientation[3]) - (2.0f * orientation[0] * orientation[1]),
                         (2.0f * orientation[0] * orientation[0]) + (2.0f * orientation[3] * orientation[3]) - 1.0f);
    }
    
    void readOrientation(float (&orientation)[4])
    {
        // Copy orientation into buffer, guarding with mutex
        {
            std::lock_guard<std::mutex> guard(m_OrientationMutex);
            std::copy_n(&m_Orientation[0], 4, &orientation[0]);
        }
    }
    
private:
    //---------------------------------------------------------------------
    // Private methods
    //---------------------------------------------------------------------
    void calculateMadgwick(float a_x, float a_y, float a_z,
                           float w_x, float w_y, float w_z)
    {
        const float deltat = 1.0f / 100.0f; // 100Hz
        const float gyroMeasError = 3.14159265358979f * (5.0f / 180.0f); // gyroscope measurement error in rad/s (shown as 5 deg/s)
        const float beta = sqrt(3.0f / 4.0f) * gyroMeasError;
        
        // Auxiliary variables to avoid reapeated calcualtions
        const float halfSEq_1 = 0.5f * m_Orientation[0];
        const float halfSEq_2 = 0.5f * m_Orientation[1];
        const float halfSEq_3 = 0.5f * m_Orientation[2];
        const float halfSEq_4 = 0.5f * m_Orientation[3];
        const float twoSEq_1 = 2.0f * m_Orientation[0];
        const float twoSEq_2 = 2.0f * m_Orientation[1];
        const float twoSEq_3 = 2.0f * m_Orientation[2];
        
        // Normalise the accelerometer measurement
        const float accelNorm = sqrt(a_x * a_x + a_y * a_y + a_z * a_z);
        a_x /= accelNorm;
        a_y /= accelNorm;
        a_z /= accelNorm;
        
        // Compute the objective function and Jacobian
        const float f_1 = twoSEq_2 * m_Orientation[3] - twoSEq_1 * m_Orientation[2] - a_x;
        const float f_2 = twoSEq_1 * m_Orientation[1] + twoSEq_3 * m_Orientation[3] - a_y;
        const float f_3 = 1.0f - twoSEq_2 * m_Orientation[1] - twoSEq_3 * m_Orientation[2] - a_z;
        const float J_11or24 = twoSEq_3;                                                    // J_11 negated in matrix multiplication
        const float J_12or23 = 2.0f * m_Orientation[3];
        const float J_13or22 = twoSEq_1;                                                    // J_12 negated in matrix multiplication
        const float J_14or21 = twoSEq_2;
        const float J_32 = 2.0f * J_14or21;                                                 // negated in matrix multiplication
        const float J_33 = 2.0f * J_11or24;                                                 // negated in matrix multiplication
        
        // Compute the gradient (matrix multiplication)
        float SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1;
        float SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3;
        float SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1;
        float SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2;
        
        // Normalise the gradient
        const float gradNorm = sqrt(SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4);
        SEqHatDot_1 /= gradNorm;
        SEqHatDot_2 /= gradNorm;
        SEqHatDot_3 /= gradNorm;
        SEqHatDot_4 /= gradNorm;
        
        // Compute the quaternion derrivative measured by gyroscopes
        const float SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z;
        const float SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y;
        const float SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x;
        const float SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x;
        
        // Compute then integrate the estimated quaternion derrivative
        m_Orientation[0] += (SEqDot_omega_1 - (beta * SEqHatDot_1)) * deltat;
        m_Orientation[1] += (SEqDot_omega_2 - (beta * SEqHatDot_2)) * deltat;
        m_Orientation[2] += (SEqDot_omega_3 - (beta * SEqHatDot_3)) * deltat;
        m_Orientation[3] += (SEqDot_omega_4 - (beta * SEqHatDot_4)) * deltat;
        
        // Normalise quaternion
        const float quatNorm = sqrt((m_Orientation[0] * m_Orientation[0]) + 
                                    (m_Orientation[1] * m_Orientation[1]) + 
                                    (m_Orientation[2] * m_Orientation[2]) + 
                                    (m_Orientation[3] * m_Orientation[3]));
        m_Orientation[0] /= quatNorm;
        m_Orientation[1] /= quatNorm;
        m_Orientation[2] /= quatNorm;
        m_Orientation[3] /= quatNorm;
    }
    
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
        
        // 0.977 degrees per ADC tick, 88Hz sampling rate,
        const float gyroToRadiansSec = 0.977f * 0.017453293f / 88.0f ;
        const float normalised = 1.0f / 511.5f;
        
        // Read data until quit flag is set
        uint16_t lastCount = 0;
        for(unsigned int f = 0; !m_ShouldQuit; f++) {
            // Read count 
            uint16_t count = 0;
            if(!readBytes(reinterpret_cast<char*>(&count), sizeof(uint16_t))) {
                std::cerr << "Cannot read count from UART" << std::endl;
                break;
            }
            // Flip endianess of count
            count = ntohs(count);
            
            // Check sequence is continuous
            if(f != 0 && ((count == 0 && lastCount == 32767) || count != (lastCount + 1))) {
                std::cerr << "Sequence error after " << f << " frames (" << lastCount << ")" << std::endl;
                break;
            }
            lastCount = count;

            // Read raw data 
            uint16_t rawData[ChannelMax];
            if(!readBytes(reinterpret_cast<char*>(&rawData[0]), sizeof(uint16_t) * ChannelMax)) {
                std::cerr << "Cannot read channel data from UART" << std::endl;
                break;
            }
            
            // Transform raw data into (hopefully) meaningful floating point values
            {
                std::lock_guard<std::mutex> guard(m_DataMutex);
                std::transform(&rawData[ChannelAccelX], &rawData[ChannelAccelMax], &m_Data[ChannelAccelX],
                               [normalised](uint16_t raw)
                               {
                                   // Swap endianess
                                   // **NOTE** 'network order' is big-endian like Atomic IMU
                                   const uint16_t host = ntohs(raw);
                                   
                                   // Convert to float and normalise
                                   return ((float)host - 511.5f) * normalised;
                               });
                
                std::transform(&rawData[ChannelGyroPitch], &rawData[ChannelGyroMax], &m_Data[ChannelGyroPitch],
                               [gyroToRadiansSec](uint16_t raw)
                               {
                                   // Swap endianess
                                   // **NOTE** 'network order' is big-endian like Atomic IMU
                                   const uint16_t host = ntohs(raw);
                                   
                                   // Convert to float and scale into radians per second
                                   return ((float)host - 511.5f) * gyroToRadiansSec;
                               });
            }
            
            // Apply Madgwick filter to calculate orientation
            {
                std::lock_guard<std::mutex> guard(m_OrientationMutex);
                calculateMadgwick(m_Data[ChannelAccelX], m_Data[ChannelAccelY], m_Data[ChannelAccelZ],
                                  m_Data[ChannelGyroPitch], m_Data[ChannelGyroRoll], m_Data[ChannelGyroYaw]);
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
    float m_Data[ChannelMax];
    
    std::mutex m_OrientationMutex;
    float m_Orientation[4];
    
    int m_UART;
};