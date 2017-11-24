#pragma once

// Forward declarations
class MotorI2C;

// Functions
void buildConnectivity();
void driveMotorFromCPU1(MotorI2C &motor, float steerThreshold = 0.1f, bool displaySteering = false);