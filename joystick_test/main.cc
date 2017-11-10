#include "../common/joystick.h"

int main()
{
    JoystickXbox360 test;

    while(true) {
        if(test.isButtonDown(JoystickXbox360::Button::DPadLeft)) {
            std::cout << "TANK LEFT" << std::endl;
        }
        else if(test.isButtonDown(JoystickXbox360::Button::DPadRight)) {
            std::cout << "TANK RIGHT" << std::endl;
        }
        else if(test.isButtonDown(JoystickXbox360::Button::DPadUp)) {
            std::cout << "TANK FORWARD" << std::endl;
        }
        else if(test.isButtonDown(JoystickXbox360::Button::DPadDown)) {
            std::cout << "TANK BACK" << std::endl;
        }

    }
    return 0;
}