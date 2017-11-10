// OpenCV includes
#include <opencv2/superres/optical_flow.hpp>

// Common includes
#include "../common/camera_360.h"
#include "../common/joystick.h"
#include "../common/lm9ds1_imu.h"
#include "../common/motor_i2c.h"

// GeNN generated code includes
#include "stone_cx_CODE/definitions.h"

// Model includes
#include "parameters.h"
#include "simulatorCommon.h"

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
constexpr double pi = 3.141592653589793238462643383279502884;

void buildOpticalFlowFilter(cv::Mat &filter, float preferredAngle) {
    // Loop through columns
    for(unsigned int x = 0; x < filter.cols; x++) {
        // Convert column to angle
        const float th = (((float)x / (float)filter.cols) * 2.0f * pi) - pi;

        // Write filter with sin of angle
        filter.at<float>(0, x) = sin(th - preferredAngle);
    }
}
}   // Anonymous namespace


int main(int argc, char *argv[])
{
    constexpr float joystickDeadzone = 0.25f;
    constexpr float velocityScale = 1.0f / 500.0f;
    constexpr float motorSteerThreshold = 2.0f;
    
    // Create joystick interface
    Joystick joystick;
    
    // Create IMU interface
    LM9DS1 imu;
    
    // Initialise IMU magnetometer
    LM9DS1::MagnetoSettings magSettings;
    imu.initMagneto(magSettings);
    
    // Create motor interface
    MotorI2C motor;
    
    // Create 360 degree camera interface
    const unsigned int cameraDevice = (argc > 1) ? std::atoi(argv[1]) : 0;
    Camera360 camera(cameraDevice, cv::Size(640, 480), cv::Size(90, 10),
                     0.5, 0.416, 0.173, 0.377, -pi);
    
    // Create two grayscale frames to hold optical flow
    cv::Mat frames[2];
    frames[0].create(10, 90, CV_8UC1);
    frames[1].create(10, 90, CV_8UC1);

    // Build a velocity filter whose preferred angle is going straighj
    cv::Mat velocityFilter(1, 90, CV_32FC1);
    buildOpticalFlowFilter(velocityFilter, 0.0f);

    cv::Ptr<cv::superres::FarnebackOpticalFlow> opticalFlow = cv::superres::createOptFlow_Farneback();
    cv::Mat flowX;
    cv::Mat flowY;
    cv::Mat flowXSum(1, 90, CV_32FC1);
    cv::Mat flowSum(1, 1, CV_32FC1);
    
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
    for(unsigned int i = 0; !joystick.isButtonDown(1); i++) {
        // Read camera frame
        if(!camera.read()) {
            return EXIT_FAILURE;
        }
        
        // Convert frame to grayscale and store in array
        const unsigned int currentFrame = i % 2;
        cv::cvtColor(camera.getUnwrappedImage(), frames[currentFrame], CV_BGR2GRAY);
        
        // Read magnetometer sample
        while(!imu.isMagnetoAvailable()) {
        }
        float magnetoData[3];
        imu.readMagneto(magnetoData);
        
        // If this isn't the first frame
        if(i > 0) {
            // Calculate optical flow
            const unsigned int prevFrame = (i - 1) % 2;
            opticalFlow->calc(frames[prevFrame], frames[currentFrame], flowX, flowY);

            // Reduce horizontal flow - summing along columns
            cv::reduce(flowX, flowXSum, 0, CV_REDUCE_SUM);

            // Multiply summed flow by filters
            cv::multiply(flowXSum, velocityFilter, flowXSum);

            // Reduce filtered flow - summing along rows
            cv::reduce(flowXSum, flowSum, 1, CV_REDUCE_SUM);
    
            // Calculate speed and apply to both TN2 hemispheres
            // **NOTE** robot is incapable of holonomic motion!
            const float speed = flowSum.at<float>(0, 0) * velocityScale;
            speedTN2[Parameters::HemisphereLeft] = speed;
            speedTN2[Parameters::HemisphereRight] = speed;
            //std::cout << "Speed:" << speed << std::endl;
            
            // Calculate heading angle from magneto data and pass to TL neurons
            headingAngleTL = atan2(magnetoData[0], magnetoData[2]);
            
            //std::cout << "Heading:" << headingAngleTL << std::endl;
            
             // Step network
            stepTimeCPU();
            
            // If we are going outbound
            if(outbound) {
                // Read joystick axis state and drive robot manually
                const float joystickX = joystick.getAxisState(0);
                const float joystickY = joystick.getAxisState(1);
                if(joystickX < -joystickDeadzone) {
                    std::cout << "left" << std::endl;
                    motor.tank(1.0f, -1.0f);
                }
                else if(joystickX > joystickDeadzone) {
                    std::cout << "right" << std::endl;
                    motor.tank(-1.0f, 1.0f);
                }
                else if(joystickY < -joystickDeadzone) {
                    std::cout << "fwd" << std::endl;
                    motor.tank(1.0f, 1.0f);
                }
                else if(joystickY > joystickDeadzone) {
                    std::cout << "back" << std::endl;
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
        
    }
    
    // Stop motor
    motor.tank(0.0f, 0.0f);
    
    // Exit
    return EXIT_SUCCESS;
}