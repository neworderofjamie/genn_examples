// Standard C++ includes
#include <chrono>
#include <numeric>

// Common includes
#include "../common/analogue_csv_recorder.h"
#include "../common/joystick.h"
#include "../common/motor_i2c.h"
#include "../common/vicon_udp.h"

// GeNN generated code includes
#include "stone_cx_CODE/definitions.h"

// Model includes
#include "parameters.h"
#include "robotParameters.h"
#include "simulatorCommon.h"


enum class ViconEvent : unsigned int
{
    TrialStart,
    HomeStart,
    TrialEnd,
};

int main(int argc, char *argv[])
{
    constexpr float speedScale = 1.0f / 400.0f;
    
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
        preferredAngleTL[i] = preferredAngleTL[8 + i] = (Parameters::pi / 4.0) * (double)i;
    }

    //---------------------------------------------------------------------------
    // Build connectivity
    //---------------------------------------------------------------------------
    buildConnectivity();
    
    initstone_cx();

#ifdef RECORD_ELECTROPHYS
    AnalogueCSVRecorder<scalar> tn2Recorder("tn2.csv", rTN2, Parameters::numTN2, "TN2");
    AnalogueCSVRecorder<scalar> cl1Recorder("cl1.csv", rCL1, Parameters::numCL1, "CL1");
    AnalogueCSVRecorder<scalar> tb1Recorder("tb1.csv", rTB1, Parameters::numTB1, "TB1");
    AnalogueCSVRecorder<scalar> cpu4Recorder("cpu4.csv", rCPU4, Parameters::numCPU4, "CPU4");
    AnalogueCSVRecorder<scalar> cpu1Recorder("cpu1.csv", rCPU1, Parameters::numCPU1, "CPU1");
#endif  // RECORD_ELECTROPHYS
    
    // Wait for VICON system to track some objects
    while(vicon.getNumObjects() == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "Waiting for object..." << std::endl;
    }

    std::ofstream eventStream("events.csv", std::ios_base::app);
    eventStream << "Frame number, event id" << std::endl;    
    eventStream << vicon.getFrameNumber() << "," << static_cast<unsigned int>(ViconEvent::TrialStart) << std::endl;

    // Loop until second joystick button is pressed
    bool outbound = true;
    unsigned int numTicks = 0;
    unsigned int numOverflowTicks = 0;
    int64_t totalMicroseconds = 0;
    for(;; numTicks++) {
        // Record time at start of tick
        const auto tickStartTime = std::chrono::high_resolution_clock::now();
        
        // Read from joystick
        joystick.read();
        
        // Stop if 2nd button is pressed
        if(joystick.isButtonDown(1)) {
            break;
        }
        
        // Read data from VICON system
        auto objectData = vicon.getObjectData(0);
        const auto &velocity = objectData.getVelocity();
        const auto &rotation = objectData.getRotation();
        //Vicon::Vector velocity{0.0, 0.0, 0.0};
        //Vicon::Vector rotation{0.0, 0.0, 0.0};
        
        // Calculate scalar speed and apply to both TN2 hemisphere
        // **NOTE** robot is incapable of holonomic motion!
        float speed = sqrt(std::accumulate(std::begin(velocity), std::end(velocity), 0.0f,
                                           [](float acc, double v){ return acc + (float)(v * v); }));

        speed *= speedScale;
        speedTN2[Parameters::HemisphereLeft] = speed;
        speedTN2[Parameters::HemisphereRight] = speed;


        // Get yaw from VICON and pass to TL neurons
        // **TODO** check axes and add enum
        headingAngleTL = rotation[2] + Parameters::pi;
        if(numTicks % 100 == 0) {
            std::cout <<  "Ticks:" << numTicks << ", Heading: " << headingAngleTL << ", Speed:" << speed << std::endl;
        }
        //std::cout << "Heading:" << headingAngleTL << std::endl;

        // Step network
        stepTimeCPU();

#ifdef RECORD_ELECTROPHYS
        tn2Recorder.record(numTicks);
        cl1Recorder.record(numTicks);
        tb1Recorder.record(numTicks);
        cpu4Recorder.record(numTicks);
        cpu1Recorder.record(numTicks);
#endif  // RECORD_ELECTROPHYS

        // If we are going outbound
        if(outbound) {
            // Drive motor using joystick
            joystick.driveMotor(motor, RobotParameters::joystickDeadzone);

            // If first button is pressed switch to returning home
            if(joystick.isButtonDown(0)) {
                std::cout << "Returning home!" << std::endl;
                outbound = false;
                eventStream << vicon.getFrameNumber() << "," << static_cast<unsigned int>(ViconEvent::HomeStart) << std::endl;
            }
        }
        // Otherwise we're returning home - use CPU1 output to drive motor
        else {
            driveMotorFromCPU1(motor, RobotParameters::motorSteerThreshold, (numTicks % 100) == 0);
        }
        
        // Record time at end of tick
        const auto tickEndTime = std::chrono::high_resolution_clock::now();
        
        // Calculate tick duration (in microseconds)
        const int64_t tickMicroseconds = std::chrono::duration_cast<chrono::microseconds>(tickEndTime - tickStartTime).count();
        
        // Add to total
        totalMicroseconds += tickMicroseconds;
        
        // If there is time left in tick, sleep for remainder
        if(tickMicroseconds < RobotParameters::targetTickMicroseconds) {
            std::this_thread::sleep_for(std::chrono::microseconds(RobotParameters::targetTickMicroseconds - tickMicroseconds));
        }
        // Otherwise, increment overflow counter
        else {
            numOverflowTicks++;
        }
    }
    
    // Show overflow stats
    std::cout << numOverflowTicks << "/" << numTicks << " ticks overflowed, mean tick time: " << (double)totalMicroseconds / (double)numTicks << "uS" << std::endl;
    eventStream << vicon.getFrameNumber() << "," << static_cast<unsigned int>(ViconEvent::TrialEnd) << std::endl;
    
    // Stop motor
    motor.tank(0.0f, 0.0f);
    
    // Exit
    return EXIT_SUCCESS;
}
