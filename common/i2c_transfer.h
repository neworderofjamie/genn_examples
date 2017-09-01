#pragma once

// Standard C++ includes
#include <vector>

// Standard C includes
#include <cstring>
#include <cstdio>
#include <cstdint>

// Posix includes
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>

// I2C includes
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#define MAX_LENGTH 4 // max lenght of buffer (we can change this)

class I2C_transfer
{
    public:
        //----- ctor-------------------
        I2C_transfer() {
            char *path = "/dev/i2c-1";
            this->setup(path);
        }
        //-----------------------------
        
        
        //------- READ DATA ---------------------------------------------------
        std::vector<uint8_t> read_data() {                           // reads data
            
            std::vector<uint8_t> readings_vector;
            
            
            if (read(file_i2c, buffer, 4) > MAX_LENGTH) {
                printf("failed to read from i2c bus");
            }
            else {
                readings_vector.push_back(buffer[0]);   // sensor 2602
                readings_vector.push_back(buffer[1]);   // sensor 2610
                readings_vector.push_back(buffer[2]);   // sensor 2611
                readings_vector.push_back(buffer[3]);   // sensor 2620

                //printf("data read -> 2602:%u - 2610:%u - 2611:%u - 2620:%u\n", buffer[0], buffer[1], buffer[2], buffer[3]);
            }
            return readings_vector;
        }
        //---------------------------------------------------------------------
        
        
        //----- WRITE DATA ----------------------------------------------------
        void write_data(uint8_t data_buffer[], int length) {  // writes data
            if (write(this->file_i2c, data_buffer, length) > MAX_LENGTH) {
                printf("failed to write to i2c bus \n");
            }
        }
        //---------------------------------------------------------------------

        
        
    private:
        int file_i2c;                                   // i2c file
        int slave_address = 0x29;                       // arduino's address
        uint8_t buffer[MAX_LENGTH];                     // buffer for i2c
        
        // --------------------- SETS UP THE I2C ------------------------------
        void setup(char *id) {                           // sets up i2c
            char *filename = id;

            if ((this->file_i2c = open(filename, O_RDWR)) < 0) {
                perror("error in setup "); // perror will tell the reason
                // the error is usually permission error for which we can
                // temporarily use < $sudo chmod 666 /dev/i2c-1 >
                return;
            }

            if (ioctl(file_i2c, I2C_SLAVE, this->slave_address) < 0) {
                printf("cannot talk to the slave\n");
                return;
            }
            else {
                printf("i2c successfully initialized \n");
            }
        }
        //---------------------------------------------------------------------

};

#endif 
