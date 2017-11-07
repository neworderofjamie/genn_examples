// Standard C includes
#include <cmath>

// Common includes
#include "../common/lm9ds1_imu.h"

int main()
{
    const float radiansToDegrees = 180.0f / 3.14159f;
    LM9DS1 imu;
    
    LM9DS1::GyroSettings gyroSettings;
    imu.initGyro(gyroSettings);
    
    LM9DS1::AccelSettings accelSettings;
    imu.initAccel(accelSettings);
    
    LM9DS1::MagnetoSettings magSettings;
    imu.initMagneto(magSettings);
    
    imu.calibrateAccelGyro();
    
    //if(!imu.calibrateMagneto()) {
    //    return EXIT_FAILURE;
    //}
    
    //return EXIT_SUCCESS;
    while(true) {
        // Read accelerometer sample
        while(!imu.isAccelAvailable()) {
        }
        float accelData[3];
        imu.readAccel(accelData);
        
        // Calculate roll and pitch
        const float roll = atan2(accelData[0], sqrt((accelData[2] * accelData[2]) + (accelData[1] * accelData[1])));
        const float pitch = atan2(accelData[2], sqrt((accelData[1] * accelData[1]) + (accelData[1] * accelData[1])));
        
        
        // Read magnetometer sample
        while(!imu.isMagnetoAvailable()) {
        }
        float magnetoData[3];
        imu.readMagneto(magnetoData);
            
        const float yawUncorrected = atan2(magnetoData[0], magnetoData[2]);
        //const float yaw = atan2((-magnetoData[0] * cos(roll)) + (magnetoData[1] * sin(roll)), 
        //                        (magnetoData[2] * cos(pitch)) + (magnetoData[1] * sin(pitch) * sin(roll)) + (magnetoData[1] * sin(pitch) * cos(roll)));
        std::cout << "Roll:" << roll * radiansToDegrees << ", Pitch:" << pitch * radiansToDegrees << ", Yaw:" << yawUncorrected * radiansToDegrees << std::endl;
    }
    return EXIT_SUCCESS;
}