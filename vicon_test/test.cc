#include <chrono>
#include <iostream>
#include <thread>

#include <cstdlib>

#include "../common/vicon_udp.h"

int main()
{
    Vicon::UDPClient<Vicon::ItemData> vicon(51001, 100);

    while(vicon.getNumItems() == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "Waiting for object" << std::endl;
    }

    while(true) {
        auto itemData = vicon.getItemData(0);
        const auto &translation = itemData.getTranslation();
        const auto &rotation = itemData.getRotation();

        std::cout << "Translation:" << translation[0] << ", " << translation[1] << ", " << translation[2] << std::endl;
        std::cout << "Rotation:" << rotation[0] << ", " << rotation[1] << ", " << rotation[2] << std::endl;
    }
    return EXIT_SUCCESS;
}