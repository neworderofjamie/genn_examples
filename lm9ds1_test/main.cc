// Common includes
#include "../common/lm9ds1_imu.h"

int main()
{
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
    
    /*if(!imu.setFIFOEnabled(false)) {
        return EXIT_FAILURE;
    }*/
    
    while(true) {
        int16_t data[3];
        while(!imu.isAccelAvailable()) {
        }
        
        if(!imu.readAccel(data)) {
            return EXIT_FAILURE;
        }
        
        /*while(!imu.isMagnetoAvailable()) {
        }
        
        if(!imu.readMagneto(data)) {
            return EXIT_FAILURE;
        }*/
        std::cout << data[0] << ", " << data[1] << ", " << data[2] << std::endl;
    }
    return EXIT_SUCCESS;
}