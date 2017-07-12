// Standard C++ includes
#include <chrono>
#include <thread>

#include "../common/motor.h"

int main()
{
    Motor motor("192.168.1.1", 2000);

    motor.tank(1.0, 1.0);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    motor.tank(0.0, 0.0);

    return 0;
}