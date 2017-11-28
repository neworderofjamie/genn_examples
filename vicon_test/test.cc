#include <chrono>
#include <iostream>
#include <thread>

#include <cstdlib>

#include "../common/vicon_udp.h"

int main()
{
    Vicon::UDPClient<Vicon::ObjectData> vicon(51001, 100);

    while(vicon.getNumObjects() == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "Waiting for object" << std::endl;
    }

    while(true) {
        auto objectData = vicon.getObjectData(0);
        const auto &translation = objectData.getTranslation();
        const auto &rotation = objectData.getRotation();

        std::cout << translation[0] << ", " << translation[1] << ", " << translation[2] << ", " << rotation[0] << ", " << rotation[1] << ", " << rotation[2] << std::endl;
    }
    return EXIT_SUCCESS;
}