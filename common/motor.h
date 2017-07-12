#pragma once

// Standard C++ includes

// Standard C includes
#include <cmath>
#include <cstdio>
#include <cstring>

// POSIX includes
#ifdef _WIN32
    #include <winsock2.h>
#else
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <unistd.h>
#endif

//----------------------------------------------------------------------------
// Motor
//----------------------------------------------------------------------------
class Motor
{
public:
    Motor(const std::string &address, unsigned int port)
    {
        // Create socket
        m_Socket = socket(AF_INET, SOCK_STREAM, 0);
        if(m_Socket < 0) {
            throw std::runtime_error("Cannot open socket");
        }
        
        // Create socket address structure
        sockaddr_in destAddress = {
            .sin_family = AF_INET, 
            .sin_port = htons(port), 
            .sin_addr = { .s_addr = inet_addr(address.c_str()) }};
        
        // Connect socket
        if(connect(m_Socket, reinterpret_cast<sockaddr*>(&destAddress), sizeof(destAddress)) < 0) {
            throw std::runtime_error("Cannot connect socket to " + address + ":" + std::to_string(port));
        }
    }
    
    ~Motor()
    {
        if(m_Socket > 0) {
            close(m_Socket);
        }
    }
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void tank(float left, float right)
    {
        // Clamp left and right within normalised range
        left = std::min(1.0f, std::max(-1.0f, left));
        right = std::min(1.0f, std::max(-1.0f, right));
        
        // Scale and convert to int
        int leftInt = (int)std::round(left * 100.0f);
        int rightInt = (int)std::round(right * 100.0f);
        
        // Generate command string
        char command[16];
        snprintf(command, 16, "#tnk(%d,%d)\n", leftInt, rightInt);
        
        // Write command to socket
        if(write(m_Socket, command, strlen(command)) < 0) {
            throw std::runtime_error("Cannot write to socket");
        }
    }

private:
    //----------------------------------------------------------------------------
    // Private members
    //----------------------------------------------------------------------------
    int m_Socket;
};