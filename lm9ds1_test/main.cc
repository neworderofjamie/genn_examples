// Standard C includes
#include <cmath>

// Common includes
#include "../common/lm9ds1_imu.h"

int main()
{
    const float radiansToDegrees = 180.0f / 3.14159f;
    LM9DS1 imu;
    
    LM9DS1::GyroSettings gyroSettings;
    if(!imu.initGyro(gyroSettings)) {
        return EXIT_FAILURE;
    }
    
    LM9DS1::AccelSettings accelSettings;
    if(!imu.initAccel(accelSettings)) {
        return EXIT_FAILURE;
    }
    
    LM9DS1::MagnetoSettings magSettings;
    if(!imu.initMagneto(magSettings)) {
        return EXIT_FAILURE;
    }
    
    if(!imu.calibrateAccelGyro()) {
        return EXIT_FAILURE;
    }
    
    //if(!imu.calibrateMagneto()) {
    //    return EXIT_FAILURE;
    //}
    
    //return EXIT_SUCCESS;
    while(true) {
        // Read accelerometer sample
        float accelData[3];
        while(!imu.isAccelAvailable()) {
        }
        
        if(!imu.readAccel(accelData)) {
            return EXIT_FAILURE;
        }
        
        // Calculate roll and pitch
        const float roll = atan2(accelData[0], sqrt((accelData[2] * accelData[2]) + (accelData[1] * accelData[1])));
        const float pitch = atan2(accelData[2], sqrt((accelData[1] * accelData[1]) + (accelData[1] * accelData[1])));
        
        
        // Read magnetometer sample
        float magnetoData[3];
        while(!imu.isMagnetoAvailable()) {
        }
        
        if(!imu.readMagneto(magnetoData)) {
            return EXIT_FAILURE;
        }
        
        const float yawUncorrected = atan2(magnetoData[0], magnetoData[2]);
        //const float yaw = atan2((-magnetoData[0] * cos(roll)) + (magnetoData[1] * sin(roll)), 
        //                        (magnetoData[2] * cos(pitch)) + (magnetoData[1] * sin(pitch) * sin(roll)) + (magnetoData[1] * sin(pitch) * cos(roll)));
        std::cout << "Roll:" << roll * radiansToDegrees << ", Pitch:" << pitch * radiansToDegrees << ", Yaw:" << yawUncorrected * radiansToDegrees << std::endl;
    }
    return EXIT_SUCCESS;
}