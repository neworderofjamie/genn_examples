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
        bool enableX = true;
        bool enableY = true;
        bool enableZ = true;
        GyroScale scale = GyroScale::DPS245;
        GyroSampleRate sampleRate = GyroSampleRate::Hz952;
        uint8_t bandwidth = 0;
        bool lowPowerEnable = false;
        GyroHPF hpf = GyroHPF::Disabled;
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
    {
        if(!init(path, accelGyroSlaveAddress, magnetoSlaveAddress)) {
            throw std::runtime_error("Cannot connect to IMU");
        }
    }
    
    //----------------------------------------------------------------------------
    // Public API
    //----------------------------------------------------------------------------
    bool init(const char *path = "/dev/i2c-1", int accelGyroSlaveAddress = 0x6B, int magnetoSlaveAddress = 0x1E)
    {
        // Connect to I2C devices
        if(!m_AccelGyroI2C.setup(path, accelGyroSlaveAddress)) {
            std::cerr << "Cannot connect to accelerometer/gyro device" << std::endl;
            return false;
        }
        if(!m_MagnetoI2C.setup(path, magnetoSlaveAddress)) {
            std::cerr << "Cannot connect to magneto device" << std::endl;
            return false;
        }
        
        // Read identities
        uint8_t accelGyroID;
        uint8_t magnetoID;
        if(!readAccelGyroByte(AccelGyroReg::WHO_AM_I_XG, accelGyroID)) {
            std::cerr << "Cannot read accelerometer/gyro id" << std::endl;
            return false;
        }
        if(!readMagnetoByte(MagnetoReg::WHO_AM_I, magnetoID)) {
            std::cerr << "Cannot read accelerometer/gyro id" << std::endl;
            return false;
        }
        
        // Check identities
        if(accelGyroID != AccelGyroID) {
            std::cerr << "Accelerometer/gyro has wrong ID" << std::endl;
            return false;
        }
        if(magnetoID != MagnetoID) {
            std::cerr << "Magneto has wrong ID" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool initGyro(const GyroSettings &settings)
    {
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
        if(!writeAccelGyroByte(AccelGyroReg::CTRL_REG1_G, ctrlReg1Value)) {
            std::cerr << "Cannot set gyro control register 1" << std::endl;
            return false;
        }
        
        // CTRL_REG2_G (Default value: 0x00)
        // [0][0][0][0][INT_SEL1][INT_SEL0][OUT_SEL1][OUT_SEL0]
        // INT_SEL[1:0] - INT selection configuration
        // OUT_SEL[1:0] - Out selection configuration
        if(!writeAccelGyroByte(AccelGyroReg::CTRL_REG2_G, 0)) {
            std::cerr << "Cannot set gyro control register 2" << std::endl;
            return false;
        }
        
        // CTRL_REG3_G (Default value: 0x00)
        // [LP_mode][HP_EN][0][0][HPCF3_G][HPCF2_G][HPCF1_G][HPCF0_G]
        // LP_mode - Low-power mode enable (0: disabled, 1: enabled)
        // HP_EN - HPF enable (0:disabled, 1: enabled)
        // HPCF_G[3:0] - HPF cutoff frequency
        uint8_t ctrlReg3Value = settings.lowPowerEnable ? (1 << 7) : 0;
        ctrlReg3Value |= static_cast<uint8_t>(settings.hpf);
        if(!writeAccelGyroByte(AccelGyroReg::CTRL_REG3_G, ctrlReg3Value)) {
            std::cerr << "Cannot set gyro control register 3" << std::endl;
            return false;
        }
        
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
        if(!writeAccelGyroByte(AccelGyroReg::CTRL_REG4, ctrlReg4Value)) {
            std::cerr << "Cannot set gyro control register 4" << std::endl;
            return false;
        }
        
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
        if(!writeAccelGyroByte(AccelGyroReg::ORIENT_CFG_G, orientCfgValue)) {
            std::cerr << "Cannot set gyro orient config register" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool initAccel(const AccelSettings &settings)
    {
        //	CTRL_REG5_XL (0x1F) (Default value: 0x38)
        //	[DEC_1][DEC_0][Zen_XL][Yen_XL][Zen_XL][0][0][0]
        //	DEC[0:1] - Decimation of accel data on OUT REG and FIFO.
        //		00: None, 01: 2 samples, 10: 4 samples 11: 8 samples
        //	Zen_XL - Z-axis output enabled
        //	Yen_XL - Y-axis output enabled
        //	Xen_XL - X-axis output enabled
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
        
        if(!writeAccelGyroByte(AccelGyroReg::CTRL_REG5_XL, ctrlReg5Value)) {
            std::cerr << "Cannot set accelerometer control register 5" << std::endl;
            return false;
        }
        
        // CTRL_REG6_XL (0x20) (Default value: 0x00)
        // [ODR_XL2][ODR_XL1][ODR_XL0][FS1_XL][FS0_XL][BW_SCAL_ODR][BW_XL1][BW_XL0]
        // ODR_XL[2:0] - Output data rate & power mode selection
        // FS_XL[1:0] - Full-scale selection
        // BW_SCAL_ODR - Bandwidth selection
        // BW_XL[1:0] - Anti-aliasing filter bandwidth selection
        uint8_t ctrlReg6Value = static_cast<uint8_t>(settings.sampleRate) << 5;
        ctrlReg6Value |= static_cast<uint8_t>(settings.scale) << 3;
        ctrlReg6Value |= static_cast<uint8_t>(settings.bandwidth);
        if(!writeAccelGyroByte(AccelGyroReg::CTRL_REG6_XL, ctrlReg6Value)) {
            std::cerr << "Cannot set accelerometer control register 6" << std::endl;
            return false;
        }
        
        // CTRL_REG7_XL (0x21) (Default value: 0x00)
        // [HR][DCF1][DCF0][0][0][FDS][0][HPIS1]
        // HR - High resolution mode (0: disable, 1: enable)
        // DCF[1:0] - Digital filter cutoff frequency
        // FDS - Filtered data selection
        // HPIS1 - HPF enabled for interrupt function
        if(!writeAccelGyroByte(AccelGyroReg::CTRL_REG7_XL, static_cast<uint8_t>(settings.highResBandwidth))) {
            std::cerr << "Cannot set accelerometer control register 7" << std::endl;
            return false;
        }
        
        std::cout << "Accelerometer initialised" << std::endl;
        return true;
    }
    
    
    bool initMagneto(const MagnetoSettings &settings)
    {
        // Calculate sensitivity
        static constexpr float magSensitivity[4] = {0.00014, 0.00029, 0.00043, 0.00058};
        m_MagnetoSensitivity = magSensitivity[static_cast<uint8_t>(settings.scale)];
        
        // CTRL_REG1_M (Default value: 0x10)
        // [TEMP_COMP][OM1][OM0][DO2][DO1][DO0][0][ST]
        // TEMP_COMP - Temperature compensation
        // OM[1:0] - X & Y axes op mode selection
        //	00:low-power, 01:medium performance
        //	10: high performance, 11:ultra-high performance
        // DO[2:0] - Output data rate selection
        // ST - Self-test enable
        uint8_t ctrlReg1Value = 0;
        if (settings.tempCompensationEnable) {
            ctrlReg1Value |= (1 << 7);
        }
        ctrlReg1Value |= static_cast<uint8_t>(settings.xyPerformance) << 5;
        ctrlReg1Value |= static_cast<uint8_t>(settings.sampleRate) << 2;
        if(!writeMagnetoByte(MagnetoReg::CTRL_REG1, ctrlReg1Value)) {
            std::cerr << "Cannot set magneto control register 1" << std::endl;
            return false;
        }
        
        // CTRL_REG2_M (Default value 0x00)
        // [0][FS1][FS0][0][REBOOT][SOFT_RST][0][0]
        // FS[1:0] - Full-scale configuration
        // REBOOT - Reboot memory content (0:normal, 1:reboot)
        // SOFT_RST - Reset config and user registers (0:default, 1:reset)
        const uint8_t ctrlReg2Value = static_cast<uint8_t>(settings.scale) << 5;
        if(!writeMagnetoByte(MagnetoReg::CTRL_REG2, ctrlReg2Value)) {
            std::cerr << "Cannot set magneto control register 2" << std::endl;
            return false;
        }
        
        
        // CTRL_REG3_M (Default value: 0x03)
        // [I2C_DISABLE][0][LP][0][0][SIM][MD1][MD0]
        // I2C_DISABLE - Disable I2C interace (0:enable, 1:disable)
        // LP - Low-power mode cofiguration (1:enable)
        // SIM - SPI mode selection (0:write-only, 1:read/write enable)
        // MD[1:0] - Operating mode
        //	00:continuous conversion, 01:single-conversion,
        //  10,11: Power-down
        uint8_t ctrlReg3Value = 0;
        if (settings.lowPowerEnable) {
            ctrlReg3Value |= (1 << 5);
        }
        ctrlReg3Value |= static_cast<uint8_t>(settings.operatingMode);
        if(!writeMagnetoByte(MagnetoReg::CTRL_REG3, ctrlReg3Value)) {
            std::cerr << "Cannot set magneto control register 3" << std::endl;
            return false;
        }
        
        // CTRL_REG4_M (Default value: 0x00)
        // [0][0][0][0][OMZ1][OMZ0][BLE][0]
        // OMZ[1:0] - Z-axis operative mode selection
        //	00:low-power mode, 01:medium performance
        //	10:high performance, 10:ultra-high performance
        // BLE - Big/little endian data
        const uint8_t ctrlReg4Value = static_cast<uint8_t>(settings.zPerformance) << 2;
        if(!writeMagnetoByte(MagnetoReg::CTRL_REG4, ctrlReg4Value)) {
            std::cerr << "Cannot set magneto control register 4" << std::endl;
            return false;
        }
        
        // CTRL_REG5_M (Default value: 0x00)
        // [0][BDU][0][0][0][0][0][0]
        // BDU - Block data update for magnetic data
        //	0:continuous, 1:not updated until MSB/LSB are read
        if(!writeMagnetoByte(MagnetoReg::CTRL_REG5, 0)){
            std::cerr << "Cannot set magneto control register 5" << std::endl;
            return false;
        }
        
        std::cout << "Magneto initialised" << std::endl;
        
        return true;
    }
    
    bool isAccelAvailable()
    {
        uint8_t status;
        if(readAccelGyroByte(AccelGyroReg::STATUS_REG_1, status)) {
            return (status & (1 << 0));
        }
        else {
            std::cerr << "Cannot read accelerometer status register" << std::endl;
            return false;
        }
    }
    
    bool isMagnetoAvailable(Axis axis = Axis::All)
    {
        // Read status register
        uint8_t status;
        if(readMagnetoByte(MagnetoReg::STATUS_REG, status)) {
            const uint8_t axisByte = static_cast<uint8_t>(axis);
            return ((status & (1 << axisByte)) >> axisByte);
        }
        else {
            std::cerr << "Cannot read magneto status register" << std::endl;
            return false;
        }
    }
    
    bool readAccel(int16_t (&data)[3])
    {
        return readAccelGyroData(AccelGyroReg::OUT_X_L_XL, data);
    }
    
    bool readMagneto(int16_t (&data)[3]) 
    {
        return readMagnetoData(MagnetoReg::OUT_X_L, data);
    }
    
    bool readMagneto(float (&data)[3]) 
    {
        int16_t dataInt[3];
        if(readMagnetoData(MagnetoReg::OUT_X_L, dataInt)) {
            std::transform(std::begin(dataInt), std::end(dataInt), std::begin(data),
                           [this](int16_t v){ return m_MagnetoSensitivity * (float)v; });
            return true;
        }
        else {
            return false;
        }
    }
    
    bool setFIFOEnabled(bool enabled)
    {
        uint8_t ctrlReg9Value;
        if(readAccelGyroByte(AccelGyroReg::CTRL_REG9, ctrlReg9Value)) {
            if(enabled) {
                ctrlReg9Value |= (1 << 6);
            }
            else {
                ctrlReg9Value &= ~(1 << 6);
            }
            
            return writeAccelGyroByte(AccelGyroReg::CTRL_REG9, ctrlReg9Value);
        }
        else {
            std::cerr << "Cannot read control register 9" << std::endl;
            return false;
        }
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
    bool readByte(I2CInterface &interface, uint8_t address, uint8_t &byte)
    {
        if(!interface.write({address})) {
            return false;
        }
        
        uint8_t value[1];
        if(!interface.read(value)) {
            return false;
        }
        else {
            byte = value[0];
            return true;
        }
    }
    
    template<typename T, size_t N>
    bool readData(I2CInterface &interface, uint8_t address, T (&data)[N])
    {
        if(!interface.write({address | 0x80})) {
            return false;
        }
        
        return interface.read(data);
    }
    
    bool writeByte(I2CInterface &interface, uint8_t address, uint8_t byte)
    {
        if(!interface.write({address})) {
            return false;
        }
        
        if(!interface.write({byte})) {
            return false;
        }
        
        uint8_t test;
        if(!readByte(interface, address, test)) {
            return false;
        }
        else if(test != byte) {
            std::cerr << "Value " << (int)byte << " not set (" << (int)test << ")" << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
    
    bool readAccelGyroByte(AccelGyroReg reg, uint8_t &byte)
    {
        return readByte(m_AccelGyroI2C, static_cast<uint8_t>(reg), byte);
    }
    
    template<typename T, size_t N>
    bool readMagnetoData(MagnetoReg reg, T (&data)[N])
    {
        return readData(m_MagnetoI2C, static_cast<uint8_t>(reg), data);
    }
    
    template<typename T, size_t N>
    bool readAccelGyroData(AccelGyroReg reg,  T (&data)[N])
    {
        return readData(m_AccelGyroI2C, static_cast<uint8_t>(reg), data);
    }
    
    bool readMagnetoByte(MagnetoReg reg, uint8_t &byte)
    {
        return readByte(m_MagnetoI2C, static_cast<uint8_t>(reg), byte);
    }
    
    bool writeAccelGyroByte(AccelGyroReg reg, uint8_t byte)
    {
        return writeByte(m_AccelGyroI2C, static_cast<uint8_t>(reg), byte);
    }
    
    bool writeMagnetoByte(MagnetoReg reg, uint8_t byte)
    {
        return writeByte(m_MagnetoI2C, static_cast<uint8_t>(reg), byte);
    }
    
    //----------------------------------------------------------------------------
    // Members
    //----------------------------------------------------------------------------
    float m_MagnetoSensitivity;
    float m_AccelSensitivity;
    I2CInterface m_AccelGyroI2C;
    I2CInterface m_MagnetoI2C;
};