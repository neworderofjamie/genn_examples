// Standard C++ includes
#include <atomic>
#include <chrono>
#include <thread>

// OpenCV includes
#include <opencv2/superres/optical_flow.hpp>

// Common includes
#include "../common/camera_360.h"
#include "../common/joystick.h"
#include "../common/lm9ds1_imu.h"
#include "../common/motor_i2c.h"
#include "../common/timer.h"

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

void imuThreadFunc(std::atomic<bool> &shouldQuit, std::atomic<float> &heading, unsigned int &numSamples)
{
    // Create IMU interface
    LM9DS1 imu;
    
    // Initialise IMU magnetometer
    LM9DS1::MagnetoSettings magSettings;
    imu.initMagneto(magSettings);
    
    // While quit signal isn't set
    for(numSamples = 0; !shouldQuit; numSamples++) {
        // Wait for magneto to become available
        while(!imu.isMagnetoAvailable()){
        }
        
        // Read magneto
        float magnetoData[3];
        imu.readMagneto(magnetoData);
            
        // Calculate heading angle from magneto data and set atomic value
        heading = atan2(magnetoData[0], magnetoData[2]);
    }
}

void opticalFlowThreadFunc(int cameraDevice, std::atomic<bool> &shouldQuit, std::atomic<float> &speed, unsigned int &numFrames)
{
    Camera360 camera(cameraDevice, cv::Size(640, 480), cv::Size(90, 10),
                     0.5, 0.416, 0.173, 0.377, -pi);
    
    // Create two grayscale frames to hold optical flow
    cv::Mat frames[2];
    frames[0].create(10, 90, CV_8UC1);
    frames[1].create(10, 90, CV_8UC1);

    // Build a velocity filter whose preferred angle is going straight (0 degrees)
    cv::Mat velocityFilter(1, 90, CV_32FC1);
    buildOpticalFlowFilter(velocityFilter, 0.0f);

    cv::Ptr<cv::superres::FarnebackOpticalFlow> opticalFlow = cv::superres::createOptFlow_Farneback();
    cv::Mat flowX;
    cv::Mat flowY;
    cv::Mat flowXSum(1, 90, CV_32FC1);
    cv::Mat flowSum(1, 1, CV_32FC1);
    
    // Read frames until should quit
    for(numFrames = 0; !shouldQuit; numFrames++) {
        if(!camera.read()) {
            std::cerr << "Cannot read from camera" << std::endl;
        }
        
        // Convert frame to grayscale and store in array
        const unsigned int currentFrame = numFrames % 2;
        cv::cvtColor(camera.getUnwrappedImage(), frames[currentFrame], CV_BGR2GRAY);
       
        // If this isn't the first frame
        if(numFrames > 0) {
            // Calculate optical flow
            const unsigned int prevFrame = (numFrames - 1) % 2;
            opticalFlow->calc(frames[prevFrame], frames[currentFrame], flowX, flowY);

            // Reduce horizontal flow - summing along columns
            cv::reduce(flowX, flowXSum, 0, CV_REDUCE_SUM);

            // Multiply summed flow by filters
            cv::multiply(flowXSum, velocityFilter, flowXSum);

            // Reduce filtered flow - summing along rows
            cv::reduce(flowXSum, flowSum, 1, CV_REDUCE_SUM);
    
            // Calculate speed and set atomic value
            speed = flowSum.at<float>(0, 0);
        }
    }
    
}
}   // Anonymous namespace


int main(int argc, char *argv[])
{
    constexpr float joystickDeadzone = 0.25f;
    constexpr float velocityScale = 1.0f / 500.0f;
    constexpr float motorSteerThreshold = 2.0f;
    constexpr int64_t targetTickMicroseconds = (int64_t)(DT * 1000.0) - 10;
    
    // Create joystick interface
    Joystick joystick;
    
    // Create motor interface
    MotorI2C motor;
    
    
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
    
    // Atomic flag for quitting child threads
    std::atomic<bool> shouldQuit{false};
    
    // Create thread to read from IMU
    unsigned int numIMUSamples = 0;
    std::atomic<float> imuHeading{0.0f};
    std::thread imuThread(&imuThreadFunc, 
                          std::ref(shouldQuit), std::ref(imuHeading), std::ref(numIMUSamples));
    
    // Create thread to calculate optical flow from camera device
    unsigned int numCameraFrames = 0;
    std::atomic<float> opticalFlowSpeed{0.0f};
    std::thread opticalFlowThread(&opticalFlowThreadFunc, (argc > 1) ? std::atoi(argv[1]) : 0, 
                                  std::ref(shouldQuit), std::ref(opticalFlowSpeed), std::ref(numCameraFrames));
    
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
        
        // Update heading from IMU
        headingAngleTL = imuHeading;
        
        // Update speed from IMU
        // **NOTE** robot is incapable of holonomic motion!
        speedTN2[Parameters::HemisphereLeft] = speedTN2[Parameters::HemisphereRight] = (opticalFlowSpeed * velocityScale);
        
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
        
        // Record time at end of tick
        const auto tickEndTime = std::chrono::high_resolution_clock::now();
        
        // Calculate tick duration (in microseconds)
        const int64_t tickMicroseconds = std::chrono::duration_cast<chrono::microseconds>(tickEndTime - tickStartTime).count();
        
        // Add to total
        totalMicroseconds += tickMicroseconds;
        
        // If there is time left in tick, sleep for remainder
        if(tickMicroseconds < targetTickMicroseconds) {
            std::this_thread::sleep_for(std::chrono::microseconds(targetTickMicroseconds - tickMicroseconds));
        }
        // Otherwise, increment overflow counter
        else {
            numOverflowTicks++;
        }
    }
    
    // Set quit flag and wait for child threads to complete
    shouldQuit = true;
    imuThread.join();
    opticalFlowThread.join();
    
    // Show stats
    std::cout << numOverflowTicks << "/" << numTicks << " ticks overflowed, mean tick time: " << (double)totalMicroseconds / (double)numTicks << "uS, ";
    std::cout << "IMU sample rate: " << (double)numIMUSamples / ((double)numTicks * DT * 0.001) << "Hz, ";
    std::cout << "Camera frame rate: " << (double)numCameraFrames / ((double)numTicks * DT * 0.001) << "FPS" << std::endl;
    
    // Stop motor
    motor.tank(0.0f, 0.0f);
    
    // Exit
    return EXIT_SUCCESS;
}