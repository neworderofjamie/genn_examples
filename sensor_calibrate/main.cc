#include <numeric>
#include <fstream>
#include <iostream>

#include <cmath>

// OpenCV includes
#include <opencv2/superres/optical_flow.hpp>

// Common includes
#include "../common/camera_360.h"
#include "../common/joystick.h"
#include "../common/lm9ds1_imu.h"
#include "../common/motor_i2c.h"
#include "../common/vicon_udp.h"

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
} // Anonymous namespace

int main(int argc, char *argv[])
{
    constexpr float joystickDeadzone = 0.25f;

    // Create joystick interface
    Joystick joystick;

    // Create motor interface
    MotorI2C motor;

    // Create VICON UDP interface
    Vicon::UDPClient<Vicon::ObjectDataVelocity> vicon(51001, 100);

    const int cameraDevice =  (argc > 1) ? std::atoi(argv[1]) : 0;
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

    // Create IMU interface
    LM9DS1 imu;
    
    // Initialise IMU magnetometer
    LM9DS1::MagnetoSettings magSettings;
    imu.initMagneto(magSettings);

    // Wait for VICON system to track some objects
    while(vicon.getNumObjects() == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << "Waiting for object..." << std::endl;
    }

    std::ofstream output("data.csv");

    output << "Frame, Vicon TX, Vicon TY, Vicon TZ, Vicon RX, Vicon RY, Ricon RZ,  Magneto Angle, Optical flow speed" << std::endl;

    for(unsigned int i = 0;;i++) {
        // Read from joystick
        joystick.read();

        // Stop if 2nd button is pressed
        if(joystick.isButtonDown(1)) {
            break;
        }

        if(!camera.read()) {
            std::cerr << "Cannot read from camera" << std::endl;
        }

        // Convert frame to grayscale and store in array
        const unsigned int currentFrame = i % 2;
        cv::cvtColor(camera.getUnwrappedImage(), frames[currentFrame], CV_BGR2GRAY);

        // Read data from VICON system
        auto objectData = vicon.getObjectData(0);
        const auto &velocity = objectData.getVelocity();
        const auto &translation = objectData.getTranslation();
        const auto &rotation = objectData.getRotation();

        // Calculate scalar speed and apply to both TN2 hemisphere
        // **NOTE** robot is incapable of holonomic motion!
        const float viconSpeed = sqrt(std::accumulate(std::begin(velocity), std::end(velocity), 0.0f,
                                                      [](float acc, double v){ return acc + (float)(v * v); }));

        // Wait for magneto to become available
        while(!imu.isMagnetoAvailable()){
        }
        
        // Read magneto
        float magnetoData[3];
        imu.readMagneto(magnetoData);
            
        // Calculate heading angle from magneto data and set atomic value
        const float magnetoHeading = atan2(magnetoData[0], magnetoData[2]);
        
        // If this isn't the first frame
        float cameraSpeed = 0.0f;
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

            // Calculate speed and set atomic value
            cameraSpeed = flowSum.at<float>(0, 0);
        }
        
        output << vicon.getFrameNumber() << "," << translation[0] << "," << translation[1] << "," << translation[2] << ",";
	output << rotation[0] << "," << rotation[1] << "," << rotation[2] << "," << magnetoHeading << "," << cameraSpeed << std::endl;

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
    }

    // Stop motor
    motor.tank(0.0f, 0.0f);

    return EXIT_SUCCESS;
}
