#pragma once

// Standard C++ includes
#include <algorithm>
#include <iostream>

// Standard C includes
#include <cstdint>

// Common includes
#include "i2c_interface.h"

//----------------------------------------------------------------------------
// LM9DS1
//----------------------------------------------------------------------------
class LM9DS1
{
public:
    //----------------------------------------------------------------------------
    // Enumerations
    //----------------------------------------------------------------------------
    enum class Axis : uint8_t
    {
        X   = 0,
        Y   = 1,
        Z   = 2,
        All = 3,
    };
    
    enum class GyroScale : uint8_t
    {
        DPS245  = 0,
        DPS500  = 1,
        DPS2000 = 3,
    };
    
    enum class GyroSampleRate : uint8_t
    {
        Disabled    = 0,
        Hz14_9      = 1,
        Hz59_5      = 2,
        Hz119       = 3,
        Hz238       = 4,
        Hz476       = 5,
        Hz952       = 6,
    };
    
    enum class GyroHPF : uint8_t
    {
        Disabled    = 0,
        Cutoff0     = (1 << 6) | 0,
        Cutoff1     = (1 << 6) | 1,
        Cutoff2     = (1 << 6) | 2,
        Cutoff3     = (1 << 6) | 3,
        Cutoff4     = (1 << 6) | 4,
        Cutoff5     = (1 << 6) | 5,
        Cutoff6     = (1 << 6) | 6,
        Cutoff7     = (1 << 6) | 7,
        Cutoff8     = (1 << 6) | 8,
        Cutoff9     = (1 << 6) | 9,
    };
    
    enum class AccelScale : uint8_t
    {
        G2  = 0,
        G16 = 1,
        G4  = 2,
        G8  = 3,
    };
    
    enum class AccelSampleRate : uint8_t
    {
        Disabled    = 0,
        Hz10        = 1,
        Hz50        = 2,
        Hz119       = 3,
        Hz238       = 4,
        Hz476       = 5,
        Hz952       = 6,
    };
    
    enum class AccelBandwidth : uint8_t
    {
        DeterminedBySampleRate  = 0,
        Hz408                   = (1 << 2) | 0,
        Hz211                   = (1 << 2) | 1,
        Hz105                   = (1 << 2) | 2,
        Hz50                    = (1 << 2) | 3,
    };
    
    enum class AccelHighResBandwidth : uint8_t
    {
        Disabled    = 0,
        ODR50       = (1 << 7) | (0 << 5),
        ODR100      = (1 << 7) | (1 << 5),
        ODR9        = (1 << 7) | (2 << 5),
        ODR400      = (1 << 7) | (3 << 5),
    };
    
    enum class MagnetoScale : uint8_t
    {
        GS4     = 0,
        GS8     = 1,
        GS12    = 2,
        GS16    = 3,
    };
    
    enum class MagnetoSampleRate : uint8_t
    {
        Hz0_625     = 0,
        Hz1_25      = 1,
        Hz2_5       = 2,
        Hz5         = 3,
        Hz10        = 4,
        Hz20        = 5,
        Hz40        = 6,
        Hz80        = 7,
    };
    
    enum class MagnetoPerformance : uint8_t
    {
        LowPower                = 0,
        MediumPerformance       = 1,
        HighPerformance         = 2,
        UltraHighPerformance    = 3,
        
    };
    
    enum class MagnetoOperatingMode : uint8_t
    {
        ContinuousConversion    = 0,
        SingleConversion        = 1,
        PowerDown               = 2,
    };
    
    //----------------------------------------------------------------------------
    // AccelSettings
    //----------------------------------------------------------------------------
    struct GyroSettings
    {
        // Which axes are enabled
        bool enableX = true;
        bool enableY = true;
        bool enableZ = true;
        GyroScale scale = GyroScale::DPS245;
        GyroSampleRate sampleRate = GyroSampleRate::Hz952;
        uint8_t bandwidth = 0;
        bool lowPowerEnable = false;
        GyroHPF hpf = GyroHPF::Disabled;
        
        // Which axes are flipped
        bool flipX = false;
        bool flipY = false;
        bool flipZ = false;
        bool latchInterrupt = true;
    };
    
    //----------------------------------------------------------------------------
    // AccelSettings
    //----------------------------------------------------------------------------
    struct AccelSettings
    {
        // Which axes are enabled
        bool enableX = true;
        bool enableY = true;
        bool enableZ = true;
        AccelScale scale = AccelScale::G2;
        AccelSampleRate sampleRate = AccelSampleRate::Hz952;
        AccelBandwidth bandwidth = AccelBandwidth::Hz50;
        AccelHighResBandwidth highResBandwidth = AccelHighResBandwidth::ODR50;
    };
    
    //----------------------------------------------------------------------------
    // MagnetoSettings
    //----------------------------------------------------------------------------
    struct MagnetoSettings
    {
        MagnetoScale scale = MagnetoScale::GS4;
        MagnetoSampleRate sampleRate = MagnetoSampleRate::Hz80;
        bool tempCompensationEnable = false;
        MagnetoPerformance xyPerformance = MagnetoPerformance::UltraHighPerformance;
        MagnetoPerformance zPerformance = MagnetoPerformance::UltraHighPerformance;
        bool lowPowerEnable = false;
        MagnetoOperatingMode operatingMode = MagnetoOperatingMode::ContinuousConversion;
    };
    
    
    LM9DS1(const char *path = "/dev/i2c-1", int accelGyroSlaveAddress = 0x6B, int magnetoSlaveAddress = 0x1E)
        : m_MagnetoSensitivity(1.0f), m_AccelSensitivity(1.0f), m_GyroSensitivity(1.0f), 
          m_MagnetoBias{0, 0, 0}, m_AccelBias{0, 0, 0}, m_GyroBias{0, 0, 0}
    {
        init(path, accelGyroSlaveAddress, magnetoSlaveAddress);
    }
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    void init(const char *path = "/dev/i2c-1", int accelGyroSlaveAddress = 0x6B, int magnetoSlaveAddress = 0x1E)
    {
        // Connect to I2C devices
        if(!m_AccelGyroI2C.setup(path, accelGyroSlaveAddress)) {
            throw std::runtime_error("Cannot connect to accelerometer/gyro device");
        }
        if(!m_MagnetoI2C.setup(path, magnetoSlaveAddress)) {
            throw std::runtime_error("Cannot connect to magneto device");
        }
        
        // Read identities
        const uint8_t accelGyroID = readAccelGyroByte(AccelGyroReg::WHO_AM_I_XG);
        const uint8_t magnetoID = readMagnetoByte(MagnetoReg::WHO_AM_I);
        
        // Check identities
        if(accelGyroID != AccelGyroID) {
            throw std::runtime_error("Accelerometer/gyro has wrong ID");
        }
        if(magnetoID != MagnetoID) {
            throw std::runtime_error("Magneto has wrong ID");
        }
    }
    
    void initGyro(const GyroSettings &settings)
    {
        // Cache gyro sensitivity
        m_GyroSensitivity = getGyroSensitivity(settings.scale);
        
        // CTRL_REG1_G (Default value: 0x00)
        // [ODR_G2][ODR_G1][ODR_G0][FS_G1][FS_G0][0][BW_G1][BW_G0]
        // ODR_G[2:0] - Output data rate selection
        // FS_G[1:0] - Gyroscope full-scale selection
        // BW_G[1:0] - Gyroscope bandwidth selection
        
        // To disable gyro, set sample rate bits to 0. We'll only set sample
        // rate if the gyro is enabled.
        uint8_t ctrlReg1Value = static_cast<uint8_t>(settings.sampleRate) << 5;
        ctrlReg1Value |= static_cast<uint8_t>(settings.scale);
        ctrlReg1Value |= (settings.bandwidth & 0x3);
        writeAccelGyroByte(AccelGyroReg::CTRL_REG1_G, ctrlReg1Value);
        
        // CTRL_REG2_G (Default value: 0x00)
        // [0][0][0][0][INT_SEL1][INT_SEL0][OUT_SEL1][OUT_SEL0]
        // INT_SEL[1:0] - INT selection configuration
        // OUT_SEL[1:0] - Out selection configuration
        writeAccelGyroByte(AccelGyroReg::CTRL_REG2_G, 0);
        
        // CTRL_REG3_G (Default value: 0x00)
        // [LP_mode][HP_EN][0][0][HPCF3_G][HPCF2_G][HPCF1_G][HPCF0_G]
        // LP_mode - Low-power mode enable (0: disabled, 1: enabled)
        // HP_EN - HPF enable (0:disabled, 1: enabled)
        // HPCF_G[3:0] - HPF cutoff frequency
        uint8_t ctrlReg3Value = settings.lowPowerEnable ? (1 << 7) : 0;
        ctrlReg3Value |= static_cast<uint8_t>(settings.hpf);
        writeAccelGyroByte(AccelGyroReg::CTRL_REG3_G, ctrlReg3Value);
        
        // CTRL_REG4 (Default value: 0x38)
        // [0][0][Zen_G][Yen_G][Xen_G][0][LIR_XL1][4D_XL1]
        // Zen_G - Z-axis output enable (0:disable, 1:enable)
        // Yen_G - Y-axis output enable (0:disable, 1:enable)
        // Xen_G - X-axis output enable (0:disable, 1:enable)
        // LIR_XL1 - Latched interrupt (0:not latched, 1:latched)
        // 4D_XL1 - 4D option on interrupt (0:6D used, 1:4D used)
        uint8_t ctrlReg4Value = 0;
        if (settings.enableZ) {
            ctrlReg4Value |= (1 << 5);
        }
        if (settings.enableY) {
            ctrlReg4Value |= (1 << 4);
        }
        if (settings.enableX) {
            ctrlReg4Value |= (1 << 3);
        }
        if (settings.latchInterrupt) {
            ctrlReg4Value |= (1 << 1);
        }
        writeAccelGyroByte(AccelGyroReg::CTRL_REG4, ctrlReg4Value);
        
        // ORIENT_CFG_G (Default value: 0x00)
        // [0][0][SignX_G][SignY_G][SignZ_G][Orient_2][Orient_1][Orient_0]
        // SignX_G - Pitch axis (X) angular rate sign (0: positive, 1: negative)
        // Orient [2:0] - Directional user orientation selection
        uint8_t orientCfgValue = 0;
        if (settings.flipX) {
            orientCfgValue |= (1 << 5);
        }
        if (settings.flipY) {
            orientCfgValue |= (1 << 4);
        }
        if (settings.flipZ) {
            orientCfgValue |= (1 << 3);
        }
        writeAccelGyroByte(AccelGyroReg::ORIENT_CFG_G, orientCfgValue);
        
        std::cout << "Gyro initialised" << std::endl;
    }
    
    void initAccel(const AccelSettings &settings)
    {
        // Cache accelerometer sensitivity
        m_AccelSensitivity = getAccelSensitivity(settings.scale);
        
        
        // CTRL_REG5_XL (0x1F) (Default value: 0x38)
        // [DEC_1][DEC_0][Zen_XL][Yen_XL][Zen_XL][0][0][0]
        // DEC[0:1] - Decimation of accel data on OUT REG and FIFO.
        //  00: None, 01: 2 samples, 10: 4 samples 11: 8 samples
        // Zen_XL - Z-axis output enabled
        // Yen_XL - Y-axis output enabled
        // Xen_XL - X-axis output enabled
        uint8_t ctrlReg5Value = 0;
        if (settings.enableZ) {
            ctrlReg5Value |= (1 << 5);
        }
        if (settings.enableY) {
            ctrlReg5Value |= (1 << 4);
        }
        if (settings.enableX) {
            ctrlReg5Value |= (1 << 3);
        }
        
        writeAccelGyroByte(AccelGyroReg::CTRL_REG5_XL, ctrlReg5Value);
        
        // CTRL_REG6_XL (0x20) (Default value: 0x00)
        // [ODR_XL2][ODR_XL1][ODR_XL0][FS1_XL][FS0_XL][BW_SCAL_ODR][BW_XL1][BW_XL0]
        // ODR_XL[2:0] - Output data rate & power mode selection
        // FS_XL[1:0] - Full-scale selection
        // BW_SCAL_ODR - Bandwidth selection
        // BW_XL[1:0] - Anti-aliasing filter bandwidth selection
        uint8_t ctrlReg6Value = static_cast<uint8_t>(settings.sampleRate) << 5;
        ctrlReg6Value |= static_cast<uint8_t>(settings.scale) << 3;
        ctrlReg6Value |= static_cast<uint8_t>(settings.bandwidth);
        writeAccelGyroByte(AccelGyroReg::CTRL_REG6_XL, ctrlReg6Value);
        
        // CTRL_REG7_XL (0x21) (Default value: 0x00)
        // [HR][DCF1][DCF0][0][0][FDS][0][HPIS1]
        // HR - High resolution mode (0: disable, 1: enable)
        // DCF[1:0] - Digital filter cutoff frequency
        // FDS - Filtered data selection
        // HPIS1 - HPF enabled for interrupt function
        writeAccelGyroByte(AccelGyroReg::CTRL_REG7_XL, static_cast<uint8_t>(settings.highResBandwidth));
        
        std::cout << "Accelerometer initialised" << std::endl;
    }
    
    
    void initMagneto(const MagnetoSettings &settings)
    {
        // Cache magneto sensitivity
        m_MagnetoSensitivity = getMagnetoSensitivity(settings.scale);
        
        // CTRL_REG1_M (Default value: 0x10)
        // [TEMP_COMP][OM1][OM0][DO2][DO1][DO0][0][ST]
        // TEMP_COMP - Temperature compensation
        // OM[1:0] - X & Y axes op mode selection
        // 00:low-power, 01:medium performance
        // 10: high performance, 11:ultra-high performance
        // DO[2:0] - Output data rate selection
        // ST - Self-test enable
        uint8_t ctrlReg1Value = 0;
        if (settings.tempCompensationEnable) {
            ctrlReg1Value |= (1 << 7);
        }
        ctrlReg1Value |= static_cast<uint8_t>(settings.xyPerformance) << 5;
        ctrlReg1Value |= static_cast<uint8_t>(settings.sampleRate) << 2;
        writeMagnetoByte(MagnetoReg::CTRL_REG1, ctrlReg1Value);
        
        // CTRL_REG2_M (Default value 0x00)
        // [0][FS1][FS0][0][REBOOT][SOFT_RST][0][0]
        // FS[1:0] - Full-scale configuration
        // REBOOT - Reboot memory content (0:normal, 1:reboot)
        // SOFT_RST - Reset config and user registers (0:default, 1:reset)
        const uint8_t ctrlReg2Value = static_cast<uint8_t>(settings.scale) << 5;
        writeMagnetoByte(MagnetoReg::CTRL_REG2, ctrlReg2Value);
        
        // CTRL_REG3_M (Default value: 0x03)
        // [I2C_DISABLE][0][LP][0][0][SIM][MD1][MD0]
        // I2C_DISABLE - Disable I2C interace (0:enable, 1:disable)
        // LP - Low-power mode cofiguration (1:enable)
        // SIM - SPI mode selection (0:write-only, 1:read/write enable)
        // MD[1:0] - Operating mode
        // 00:continuous conversion, 01:single-conversion,
        //  10,11: Power-down
        uint8_t ctrlReg3Value = 0;
        if (settings.lowPowerEnable) {
            ctrlReg3Value |= (1 << 5);
        }
        ctrlReg3Value |= static_cast<uint8_t>(settings.operatingMode);
        writeMagnetoByte(MagnetoReg::CTRL_REG3, ctrlReg3Value);
        
        // CTRL_REG4_M (Default value: 0x00)
        // [0][0][0][0][OMZ1][OMZ0][BLE][0]
        // OMZ[1:0] - Z-axis operative mode selection
        // 00:low-power mode, 01:medium performance
        // 10:high performance, 10:ultra-high performance
        // BLE - Big/little endian data
        const uint8_t ctrlReg4Value = static_cast<uint8_t>(settings.zPerformance) << 2;
        writeMagnetoByte(MagnetoReg::CTRL_REG4, ctrlReg4Value);
        
        // CTRL_REG5_M (Default value: 0x00)
        // [0][BDU][0][0][0][0][0][0]
        // BDU - Block data update for magnetic data
        // 0:continuous, 1:not updated until MSB/LSB are read
        writeMagnetoByte(MagnetoReg::CTRL_REG5, 0);
        
        std::cout << "Magnetometer initialised" << std::endl;
    }
    
    void calibrateAccelGyro()
    {
        std::cout << "Calibrating accelerometer and gyroscope" << std::endl;
        
        setFIFOEnabled(true);
        setFIFOMode(FIFOMode::Threshold, 31);
        
        // Accumulate 32 samples
        unsigned int numSamples = 0;
        while(numSamples < 31) {
            numSamples = getNumFIFOSamples();
        }
        
        // Accumulate bias from sensor samples
        // **NOTE** 32-bit to prevent overflow
        int32_t accelBias[3] = {0, 0, 0};
        int32_t gyroBias[3] = {0, 0, 0};
        for(unsigned int i = 0; i < numSamples; i++)
        {
            // Read a sample from gyroscope
            int16_t gyroSample[3];
            readGyro(gyroSample);

            // Add to gyro bias
            gyroBias[0] += gyroSample[0];
            gyroBias[1] += gyroSample[1];
            gyroBias[2] += gyroSample[2];
            
            // Read a sample from accelerometer
            int16_t accelSample[3];
            readAccel(accelSample);
            
            // Add to acclerometer bias
            // **NOTE** we subtract gravity from Y as sensor is vertical in current robot
            accelBias[0] += accelSample[0];
            accelBias[1] += accelSample[1] - (int16_t)(1.0f / m_AccelSensitivity);
            accelBias[2] += accelSample[2];
        }
        
        //  Divide biases by number of samples to get means
        std::transform(std::begin(accelBias), std::end(accelBias), std::begin(m_AccelBias),
                       [numSamples](int32_t v){ return v / (int32_t)numSamples; });
        std::transform(std::begin(gyroBias), std::end(gyroBias), std::begin(m_GyroBias),
                       [numSamples](int32_t v){ return v / (int32_t)numSamples; });
        
        setFIFOEnabled(false);
        setFIFOMode(FIFOMode::Off, 0);
        
        std::cout << "\tAccel bias:" << m_AccelBias[0] << "," << m_AccelBias[1] << "," << m_AccelBias[2] << std::endl;
        std::cout << "\tGyro bias:" << m_GyroBias[0] << "," << m_GyroBias[1] << "," << m_GyroBias[2] << std::endl;
    }
    
    void calibrateMagneto()
    {
        std::cout << "Calibrating magnetometer" << std::endl;
        
        int16_t magMin[3] = {0, 0, 0};
        int16_t magMax[3] = {0, 0, 0};  // The road warrior
        
        for(unsigned int i = 0; i < 128; i++) {
            // Wait for magneto data to become available
            while(!isMagnetoAvailable()) {
            }
            
            // Read sample from magneto
            int16_t magSample[3];
            readMagneto(magSample);
          
            // Update max and min
            std::transform(std::begin(magSample), std::end(magSample), std::begin(magMin), std::begin(magMin),
                           [](int16_t v, int16_t min){ return std::min(v, min); });
            std::transform(std::begin(magSample), std::end(magSample), std::begin(magMax), std::begin(magMax),
                           [](int16_t v, int16_t max){ return std::max(v, max); });
        }
        
        // Calculate bias
        int16_t magBias[3];
        std::transform(std::begin(magMin), std::end(magMin), std::begin(magMax), std::begin(magBias),
                       [](int16_t min, int16_t max){ return (min + max) / 2; });
        std::cout << "\tBias: " << magBias[0] << ", " << magBias[1] << ", " << magBias[2] << std::endl;
        
        // Set device bias
        setMagnetoOffset(MagnetoReg::OFFSET_X_REG_L, MagnetoReg::OFFSET_X_REG_H, magBias[0]);
        setMagnetoOffset(MagnetoReg::OFFSET_Y_REG_L, MagnetoReg::OFFSET_Y_REG_H, magBias[1]);
        setMagnetoOffset(MagnetoReg::OFFSET_Z_REG_L, MagnetoReg::OFFSET_Z_REG_H, magBias[2]);
    }
    
    bool isAccelAvailable()
    {
        return (readAccelGyroByte(AccelGyroReg::STATUS_REG_1) & (1 << 0));
    }
    
    bool isMagnetoAvailable(Axis axis = Axis::All)
    {
        const uint8_t axisByte = static_cast<uint8_t>(axis);
        return ((readMagnetoByte(MagnetoReg::STATUS_REG) & (1 << axisByte)) >> axisByte);
    }
    
    void readGyro(int16_t (&data)[3])
    {
        readAccelGyroData(AccelGyroReg::OUT_X_L_G, data);
        std::transform(std::begin(data), std::end(data), std::begin(m_GyroBias), std::begin(data),
                       [this](int16_t v, int16_t bias){ return v - bias; });
    }
    
    void readAccel(int16_t (&data)[3])
    {
        readAccelGyroData(AccelGyroReg::OUT_X_L_XL, data);
        std::transform(std::begin(data), std::end(data), std::begin(m_AccelBias), std::begin(data),
                       [this](int16_t v, int16_t bias){ return v - bias; });
    }
    
    void readMagneto(int16_t (&data)[3]) 
    {
        readMagnetoData(MagnetoReg::OUT_X_L, data);
    }
    
    void readGyro(float (&data)[3]) 
    {
        int16_t dataInt[3];
        readGyro(dataInt);
        std::transform(std::begin(dataInt), std::end(dataInt), std::begin(data),
                       [this](int16_t v){ return m_GyroSensitivity * (float)v; });
    }
    
    void readAccel(float (&data)[3]) 
    {
        int16_t dataInt[3];
        readAccel(dataInt);
        std::transform(std::begin(dataInt), std::end(dataInt), std::begin(data),
                       [this](int16_t v){ return m_AccelSensitivity * (float)v; });
    }
    
    void readMagneto(float (&data)[3]) 
    {
        int16_t dataInt[3];
        readMagneto(dataInt);
        std::transform(std::begin(dataInt), std::end(dataInt), std::begin(data),
                       [this](int16_t v){ return m_MagnetoSensitivity * (float)v; });
    }

private:
    //----------------------------------------------------------------------------
    // Constants
    //----------------------------------------------------------------------------
    static constexpr uint8_t AccelGyroID = 0x68;
    static constexpr uint8_t MagnetoID = 0x3D;
    
    //----------------------------------------------------------------------------
    // Enumerations
    //----------------------------------------------------------------------------
    enum class FIFOMode : uint8_t
    {
        Off                 = 0,
        Threshold           = 1,
        ContinuousTrigger   = 3,
        OffTrigger          = 4,
        Continuous          = 5,
    };
    
    enum class AccelGyroReg : uint8_t
    {
        ACT_THS = 0x04,
        ACT_DUR = 0x05,
        INT_GEN_CFG_XL = 0x06,
        INT_GEN_THS_X_XL = 0x07,
        INT_GEN_THS_Y_XL = 0x08,
        INT_GEN_THS_Z_XL = 0x09,
        INT_GEN_DUR_XL = 0x0A,
        REFERENCE_G = 0x0B,
        INT1_CTRL = 0x0C,
        INT2_CTRL = 0x0D,
        WHO_AM_I_XG = 0x0F,
        CTRL_REG1_G = 0x10,
        CTRL_REG2_G = 0x11,
        CTRL_REG3_G = 0x12,
        ORIENT_CFG_G = 0x13,
        INT_GEN_SRC_G = 0x14,
        OUT_TEMP_L = 0x15,
        OUT_TEMP_H = 0x16,
        STATUS_REG_0 = 0x17,
        OUT_X_L_G = 0x18,
        OUT_X_H_G = 0x19,
        OUT_Y_L_G = 0x1A,
        OUT_Y_H_G = 0x1B,
        OUT_Z_L_G = 0x1C,
        OUT_Z_H_G = 0x1D,
        CTRL_REG4 = 0x1E,
        CTRL_REG5_XL = 0x1F,
        CTRL_REG6_XL = 0x20,
        CTRL_REG7_XL = 0x21,
        CTRL_REG8 = 0x22,
        CTRL_REG9 = 0x23,
        CTRL_REG10 = 0x24,
        INT_GEN_SRC_XL = 0x26,
        STATUS_REG_1 = 0x27,
        OUT_X_L_XL = 0x28,
        OUT_X_H_XL = 0x29,
        OUT_Y_L_XL = 0x2A,
        OUT_Y_H_XL = 0x2B,
        OUT_Z_L_XL = 0x2C,
        OUT_Z_H_XL = 0x2D,
        FIFO_CTRL = 0x2E,
        FIFO_SRC = 0x2F,
        INT_GEN_CFG_G = 0x30,
        INT_GEN_THS_XH_G = 0x31,
        INT_GEN_THS_XL_G = 0x32,
        INT_GEN_THS_YH_G = 0x33,
        INT_GEN_THS_YL_G = 0x34,
        INT_GEN_THS_ZH_G = 0x35,
        INT_GEN_THS_ZL_G = 0x36,
        INT_GEN_DUR_G = 0x37,
    };
    
    enum class MagnetoReg : uint8_t
    {
        OFFSET_X_REG_L = 0x05,
        OFFSET_X_REG_H = 0x06,
        OFFSET_Y_REG_L = 0x07,
        OFFSET_Y_REG_H = 0x08,
        OFFSET_Z_REG_L = 0x09,
        OFFSET_Z_REG_H = 0x0A,
        WHO_AM_I = 0x0F,
        CTRL_REG1 = 0x20,
        CTRL_REG2 = 0x21,
        CTRL_REG3 = 0x22,
        CTRL_REG4 = 0x23,
        CTRL_REG5 = 0x24,
        STATUS_REG = 0x27,
        OUT_X_L = 0x28,
        OUT_X_H = 0x29,
        OUT_Y_L = 0x2A,
        OUT_Y_H = 0x2B,
        OUT_Z_L = 0x2C,
        OUT_Z_H = 0x2D,
        INT_CFG = 0x30,
        INT_SRC = 0x30,
        INT_THS_L = 0x32,
        INT_THS_H = 0x33,
    };
    
    //----------------------------------------------------------------------------
    // Private methods
    //----------------------------------------------------------------------------
    uint8_t readByte(I2CInterface &interface, uint8_t address)
    {
        if(!interface.writeByte(address)) {
            throw std::runtime_error("Cannot select read address '" + std::to_string(address) + "'");
        }
        
        uint8_t byte;
        if(interface.readByte(byte)) {
            return byte;
        }
        else {
            throw std::runtime_error("Cannot read from address '" + std::to_string(address) + "'");
        }
    }
    
    template<typename T, size_t N>
    void readData(I2CInterface &interface, uint8_t address, T (&data)[N])
    {
        if(!interface.writeByte(address | 0x80)) {
            throw std::runtime_error("Cannot select read address '" + std::to_string(address) + "'");
        }
        
        if(!interface.read(data)) {
            throw std::runtime_error("Cannot read from address '" + std::to_string(address) + "'");
        }
    }
    
    void writeByte(I2CInterface &interface, uint8_t address, uint8_t byte)
    {
        if(!interface.writeByteCommand(address, byte)) {
            throw std::runtime_error("Cannot write '" + std::to_string(byte) + "' to address '" + std::to_string(address) + "'");
        }
    }
    
    uint8_t readAccelGyroByte(AccelGyroReg reg)
    {
        return readByte(m_AccelGyroI2C, static_cast<uint8_t>(reg));
    }
    
    template<typename T, size_t N>
    void readMagnetoData(MagnetoReg reg, T (&data)[N])
    {
        readData(m_MagnetoI2C, static_cast<uint8_t>(reg), data);
    }
    
    template<typename T, size_t N>
    void readAccelGyroData(AccelGyroReg reg,  T (&data)[N])
    {
        readData(m_AccelGyroI2C, static_cast<uint8_t>(reg), data);
    }
    
    uint8_t readMagnetoByte(MagnetoReg reg)
    {
        return readByte(m_MagnetoI2C, static_cast<uint8_t>(reg));
    }
    
    void writeAccelGyroByte(AccelGyroReg reg, uint8_t byte)
    {
        writeByte(m_AccelGyroI2C, static_cast<uint8_t>(reg), byte);
    }
    
    void writeMagnetoByte(MagnetoReg reg, uint8_t byte)
    {
        writeByte(m_MagnetoI2C, static_cast<uint8_t>(reg), byte);
    }
    
    void setMagnetoOffset(MagnetoReg lowReg, MagnetoReg highReg, uint16_t axisBias)
    {
        const uint8_t axisBiasMSB = (axisBias & 0xFF00) >> 8;
        const uint8_t axisBiasLSB = (axisBias & 0x00FF);
        writeMagnetoByte(lowReg, axisBiasLSB);
        writeMagnetoByte(highReg, axisBiasMSB);
    }
    
    void setFIFOEnabled(bool enabled)
    {
        uint8_t ctrlReg9Value = readAccelGyroByte(AccelGyroReg::CTRL_REG9);
        if(enabled) {
            ctrlReg9Value |= (1 << 1);
        }
        else {
            ctrlReg9Value &= ~(1 << 1);
        }
        writeAccelGyroByte(AccelGyroReg::CTRL_REG9, ctrlReg9Value);
    }
    
    void setFIFOMode(FIFOMode mode, uint8_t threshold) {
        // Clamp threshold
        threshold = std::min((uint8_t)31, threshold);
        
        const uint8_t fifoCtrl = (static_cast<uint8_t>(mode) << 5) | threshold;
        writeAccelGyroByte(AccelGyroReg::FIFO_CTRL, fifoCtrl);
    }
    
    unsigned int getNumFIFOSamples()
    {
        return readAccelGyroByte(AccelGyroReg::FIFO_SRC) & 0x3F;
    }
    
    float getGyroSensitivity(GyroScale scale) const
    {
        switch(scale) {
            case GyroScale::DPS245:
                return 245.0f / 32768.0f;
            case GyroScale::DPS500:
                return 500.0f / 32768.0f;
            case GyroScale::DPS2000:
                return 2000.0f / 32768.0f;
        }
    }
    
    float getAccelSensitivity(AccelScale scale) const
    {
        switch(scale) {
            case AccelScale::G2:
                return 2.0f / 32768.0f;
            case AccelScale::G16:
                return 16.0f / 32768.0f;
            case AccelScale::G4:
                return 4.0f /  32768.0f;
            case AccelScale::G8:
                return 8.0f /  32768.0f;
        }
    }
    
    float getMagnetoSensitivity(MagnetoScale scale) const
    {
        switch(scale) {
            case MagnetoScale::GS4:
                return 0.00014f;
            case MagnetoScale::GS8:
                return 0.00029f;
            case MagnetoScale::GS12:
                return 0.00043f;
            case MagnetoScale::GS16:
                return 0.00058f;
        }
    }
    
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    float m_MagnetoSensitivity;
    float m_AccelSensitivity;
    float m_GyroSensitivity;
  
    int16_t m_MagnetoBias[3];
    int16_t m_AccelBias[3];
    int16_t m_GyroBias[3];
    
    I2CInterface m_AccelGyroI2C;
    I2CInterface m_MagnetoI2C;
};