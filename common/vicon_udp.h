#pragma once

// Standard C++ includes
#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Standard C includes
#include <cassert>
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
// Vicon Typedefines
//----------------------------------------------------------------------------
namespace Vicon
{
typedef double Vector[3];

//----------------------------------------------------------------------------
// Vicon::ItemData
//----------------------------------------------------------------------------
//! Simplest item data class - just tracks position and translation
class ItemData
{
public:
    ItemData() : m_Translation{0.0, 0.0, 0.0}, m_Rotation{0.0, 0.0, 0.0}
    {
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void update(const Vector &translation, const Vector &rotation)
    {
        // Copy vectors into class
        std::copy(std::begin(translation), std::end(translation), std::begin(m_Translation));
        std::copy(std::begin(rotation), std::end(rotation), std::begin(m_Rotation));
    }

    const Vector &getTranslation() const{ return m_Translation; }
    const Vector &getRotation() const{ return m_Rotation; }

private:
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    Vector m_Translation;
    Vector m_Rotation;
};

//----------------------------------------------------------------------------
// Vicon::UDPClient
//----------------------------------------------------------------------------
// Receiver for Vicon UDP streams
template<typename ItemClass>
class UDPClient
{
public:
    UDPClient(){}
    UDPClient(unsigned int port, int sampleRate)
    {
        if(!connect(port, sampleRate)) {
            throw std::runtime_error("Cannot connect");
        }
    }

    ~UDPClient()
    {
        // Set quit flag and join read thread
        m_ShouldQuit = true;
        m_ReadThread.join();
    }

    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool connect(unsigned int port, int sampleRate)
    {
        // Create socket
        int socket = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if(socket < 0) {
            std::cerr << "Cannot open socket: " << strerror(errno) << std::endl;
            return false;
        }

        // Create socket address structure
        sockaddr_in localAddress = {
            .sin_family = AF_INET,
            .sin_port = htons(port),
            .sin_addr = { .s_addr = htonl(INADDR_ANY) }};

        // Bind socket to local port
        if(bind(socket, reinterpret_cast<sockaddr*>(&localAddress), sizeof(localAddress)) < 0) {
            std::cerr << "Cannot bind socket: " << strerror(errno) << std::endl;
            return false;
        }

        // Clear atomic stop flag and start thread
        m_ShouldQuit = false;
        m_ReadThread = std::thread(&UDPClient::readThread, this, socket);
        return true;
    }

    unsigned int getNumItems()
    {
        std::lock_guard<std::mutex> guard(m_ItemDataMutex);
        return m_ItemData.size();
    }

    ItemClass getItemData(unsigned int id)
    {
        std::lock_guard<std::mutex> guard(m_ItemDataMutex);
        if(id < m_ItemData.size()) {
            return m_ItemData[id];
        }
        else {
            throw std::runtime_error("Invalid item id:" + std::to_string(id));
        }
    }

private:
    //----------------------------------------------------------------------------
    // Private API
    //----------------------------------------------------------------------------
    void updateItemData(unsigned int id, const Vector &translation, const Vector &rotation)
    {
        // Lock mutex
        std::lock_guard<std::mutex> guard(m_ItemDataMutex);

        // If no item data structure has been created for this ID, add one
        if(id >= m_ItemData.size()) {
            m_ItemData.resize(id + 1);
        }

        // Update item data with translation and rotation
        m_ItemData[id].update(translation, rotation);
    }

    void readThread(int socket)
    {
        // Create buffer for reading data
        // **NOTE** this is the maximum size supported by Vicon so will support all payload sizes
        uint8_t buffer[1024];

        // Loop until quit flag is set
        for(unsigned int f = 0; !m_ShouldQuit; f++) {
            // Read datagram
            ssize_t bytesReceived = recvfrom(socket, &buffer[0], 1024,
                                             0, NULL, NULL);

            // If there was an error stop
            if(bytesReceived < 0) {
                std::cerr << "Cannot read datagram: " << strerror(errno) << std::endl;
                break;
            }
            // Otherwise, if data was received
            else if(bytesReceived > 0) {
                // Read frame number
                uint32_t frameNumber;
                memcpy(&frameNumber, &buffer[0], sizeof(uint32_t));

                // Read items in block
                const unsigned int itemsInBlock = (unsigned int)buffer[4];

                // Loop through items in blcok
                unsigned int itemOffset = 5;
                for(unsigned int i = 0; i < itemsInBlock; i++) {
                    // Read item ID
                    const unsigned int itemID = (unsigned int)buffer[itemOffset];

                    // Read size of item
                    uint16_t itemDataSize;
                    memcpy(&itemDataSize, &buffer[itemOffset + 1], sizeof(uint16_t));
                    assert(itemDataSize == 72);

                    // Read item translation
                    double itemTranslation[3];
                    memcpy(&itemTranslation[0], &buffer[itemOffset + 27], 3 * sizeof(double));

                    // Read item rotation
                    double itemRotation[3];
                    memcpy(&itemRotation[0], &buffer[itemOffset + 51], 3 * sizeof(double));

                    // Update offset for next offet
                    itemOffset += itemDataSize;
                }
            }
        }

        // Close socket
        close(socket);
    }

    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    std::atomic<bool> m_ShouldQuit;
    std::thread m_ReadThread;

    std::mutex m_ItemDataMutex;
    std::vector<ItemClass> m_ItemData;
};
} // namespace Vicon