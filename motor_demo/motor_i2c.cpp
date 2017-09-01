// Standard C++ includes
#include <chrono>
#include <thread>

#include "../common/motor_i2c.h"

int main()
{
    MotorI2C motor;

    motor.tank(1, 1);
    std::this_thread::sleep_for(std::chrono::seconds(2));
    motor.tank(0, 0);

    return 0;
}