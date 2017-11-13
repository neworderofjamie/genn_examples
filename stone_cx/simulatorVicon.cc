// Standard C++ includes
#include <numeric>

// Common includes
#include "../common/joystick.h"
#include "../common/motor_i2c.h"
#include "../common/vicon_udp.h"

// GeNN generated code includes
#include "stone_cx_CODE/definitions.h"

// Model includes
#include "parameters.h"
#include "simulatorCommon.h"


int main(int argc, char *argv[])
{
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr float joystickDeadzone = 0.25f;
    constexpr float velocityScale = 1.0f / 500.0f;
    constexpr float motorSteerThreshold = 2.0f;
    
    // Create joystick interface
    Joystick joystick;

    // Create motor interface
    MotorI2C motor;

    // Create VICON UDP interface
    Vicon::UDPClient<Vicon::ObjectDataVelocity> vicon(51001, 100);

    // Initialise GeNN
    allocateMem();
    initialize();

    //---------------------------------------------------------------------------
    // Initialize neuron parameters
    //---------------------------------------------------------------------------
    // TL
    for(unsigned int i = 0; i < 8; i++) {
        preferredAngleTL[i] = preferredAngleTL[8 + i] = (pi / 4.0) * (double)i;
    }

    //---------------------------------------------------------------------------
    // Build connectivity
    //---------------------------------------------------------------------------
    buildConnectivity();
    
    initstone_cx();
    
    // Loop until second joystick button is pressed
    bool outbound = true;
    while(!joystick.isButtonDown(1)) {
        // Read data from VICON system
        auto objectData = vicon.getObjectData(0);
        const auto &velocity = objectData.getVelocity();
        const auto &rotation = objectData.getRotation();

        // Calculate scalar speed and apply to both TN2 hemisphere
        // **NOTE** robot is incapable of holonomic motion!
        const float speed = sqrt(std::accumulate(std::begin(velocity), std::end(velocity), 0.0f,
                                                 [](float acc, double v){ return acc + (float)(v * v); }));

        speedTN2[Parameters::HemisphereLeft] = speed;
        speedTN2[Parameters::HemisphereRight] = speed;


        // Get yaw from VICON and pass to TL neurons
        // **TODO** check axes and add enum
        headingAngleTL = rotation[2];

        //std::cout << "Heading:" << headingAngleTL << std::endl;

        // Step network
        stepTimeCPU();

        // If we are going outbound
        if(outbound) {
            // Read joystick axis state and drive robot manually
            const float joystickX = joystick.getAxisState(0);
            const float joystickY = joystick.getAxisState(1);
            if(joystickX < -joystickDeadzone) {
                motor.tank(1.0f, -1.0f);
            }
            else if(joystickX > joystickDeadzone) {
                motor.tank(-1.0f, 1.0f);
            }
            else if(joystickY < -joystickDeadzone) {
                motor.tank(1.0f, 1.0f);
            }
            else if(joystickY > joystickDeadzone) {
                motor.tank(-1.0f, -1.0f);
            }
            else {
                motor.tank(0.0f, 0.0f);
            }

            // If first button is pressed switch to returning home
            if(joystick.isButtonDown(0)) {
                std::cout << "Returning home!" << std::endl;
                outbound = false;
            }
        }
        // Otherwise we're returning home
        else {
            // Sum left and right motor activity
            const scalar leftMotor = std::accumulate(&rCPU1[0], &rCPU1[8], 0.0f);
            const scalar rightMotor = std::accumulate(&rCPU1[8], &rCPU1[16], 0.0f);

            // Steer based on signal
            const scalar steering = leftMotor - rightMotor;
            if(steering > motorSteerThreshold) {
                motor.tank(1.0f, -1.0f);
            }
            else if(steering < -motorSteerThreshold) {
                motor.tank(-1.0f, 1.0f);
            }
            else {
                motor.tank(1.0f, 1.0f);
            }
        }
    }
    
    // Stop motor
    motor.tank(0.0f, 0.0f);
    
    // Exit
    return EXIT_SUCCESS;
}