#include "simulatorCommon.h"

// Standard C++ includes
#include <iostream>
#include <numeric>

// Common includes
#include "../common/connectors.h"
#include "../common/motor_i2c.h"

// GeNN generated code includes
#include "stone_cx_CODE/definitions.h"

// Model includes
#include "parameters.h"

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{
void buildTBToCPUConnector(unsigned int numPre, unsigned int numPost,
                           SparseProjection &sparseProjection, AllocateFn allocateFn)
{
    if(numPost != (2 * numPre)) {
        throw std::runtime_error("TB-to-CPU connector can only be used when the postsynaptic population is double the size of the presynaptic population");
    }

    // Allocate SparseProjection arrays
    allocateFn(numPost);

    // Configure synaptic rows
    for(unsigned int i = 0; i < numPre; i++)
    {
        sparseProjection.indInG[i] = i * 2;
        sparseProjection.ind[i * 2] = i;
        sparseProjection.ind[(i * 2) + 1] = i + 8;
    }
    sparseProjection.indInG[numPre] = numPost;
}
}   // Anonymous namespace

void buildConnectivity()
{
    // TL_CL1
    buildOneToOneConnector(Parameters::numTL, Parameters::numCL1,
                           CTL_CL1, allocateTL_CL1);

    std::cout << "TL->CL1" << std::endl;
    printSparseMatrix(Parameters::numTL, CTL_CL1);

    // CL1_TB1
    allocateCL1_TB1(Parameters::numCL1);
    std::iota(&CCL1_TB1.indInG[0], &CCL1_TB1.indInG[Parameters::numCL1 + 1], 0);
    for(unsigned int i = 0; i < Parameters::numCL1; i++) {
        CCL1_TB1.ind[i] = i % 8;
    }
    std::cout << std::endl << "CL1->TB1" << std::endl;
    printSparseMatrix(Parameters::numCL1, CCL1_TB1);

    // TB1_TB1
    for(unsigned int i = 0; i < Parameters::numTB1; i++) {
        for(unsigned int j = 0; j < Parameters::numTB1; j++) {
            const double preferredI = preferredAngleTL[i];
            const double preferredJ = preferredAngleTL[j];

            const double w = (cos(preferredI - preferredJ) - 1.0) / 2.0;
            gTB1_TB1[(i * Parameters::numTB1) + j] = Parameters::c * w;
        }
    }
    std::cout << "TB1->TB1" << std::endl;
    printDenseMatrix(Parameters::numTB1, Parameters::numTB1, gTB1_TB1);

    // CPU4_Pontine
    buildOneToOneConnector(Parameters::numCPU4, Parameters::numPontine,
                           CCPU4_Pontine, allocateCPU4_Pontine);
    std::cout << std::endl << "CPU4->Pontine" << std::endl;
    printSparseMatrix(Parameters::numCPU4, CCPU4_Pontine);

    // TB1_CPU4
    buildTBToCPUConnector(Parameters::numTB1, Parameters::numCPU4,
                          CTB1_CPU4, allocateTB1_CPU4);
    std::cout << std::endl << "TB1->CPU4" << std::endl;
    printSparseMatrix(Parameters::numTB1, CTB1_CPU4);

    // TB1_CPU1
    buildTBToCPUConnector(Parameters::numTB1, Parameters::numCPU1,
                          CTB1_CPU1, allocateTB1_CPU1);
    std::cout << std::endl << "TB1->CPU1" << std::endl;
    printSparseMatrix(Parameters::numTB1, CTB1_CPU1);

    // CPU4_CPU1
    allocateCPU4_CPU1(Parameters::numCPU4);
    std::iota(&CCPU4_CPU1.indInG[0], &CCPU4_CPU1.indInG[Parameters::numCPU4 + 1], 0);
    CCPU4_CPU1.ind[0] = 15;
    for(unsigned int i = 0; i < 7; i++) {
        CCPU4_CPU1.ind[1 + i] = 8 + i;
        CCPU4_CPU1.ind[8 + i] = 1 + i;
    }
    CCPU4_CPU1.ind[15] = 0;
    std::cout << std::endl << "CPU4->CPU1" << std::endl;
    printSparseMatrix(Parameters::numCPU4, CCPU4_CPU1);

    // TN2_CPU4
    allocateTN2_CPU4(Parameters::numCPU4);
    CTN2_CPU4.indInG[Parameters::HemisphereLeft] = 0;
    CTN2_CPU4.indInG[Parameters::HemisphereRight] = 8;
    CTN2_CPU4.indInG[Parameters::HemisphereMax] = Parameters::numCPU4;
    std::iota(&CTN2_CPU4.ind[0], &CTN2_CPU4.ind[Parameters::numCPU4], 0);
    std::cout << std::endl << "TN2->CPU4" << std::endl;
    printSparseMatrix(Parameters::numTN2, CTN2_CPU4);

    // Pontine_CPU1
    allocatePontine_CPU1(Parameters::numPontine);
    std::iota(&CPontine_CPU1.indInG[0], &CPontine_CPU1.indInG[Parameters::numPontine + 1], 0);
    for(unsigned int i = 0; i < 5; i++) {
        CPontine_CPU1.ind[i] = 11 + i;
        CPontine_CPU1.ind[i + 11] = i;
    }
    for(unsigned int i = 0; i < 3; i++) {
        CPontine_CPU1.ind[i + 5] = 8 + i;
        CPontine_CPU1.ind[i + 8] = 5 + i;
    }
    std::cout << std::endl << "Pontine->CPU1" << std::endl;
    printSparseMatrix(Parameters::numPontine, CPontine_CPU1);
}

void driveMotorFromCPU1(MotorI2C &motor, float steerThreshold, bool displaySteering)
{
    // Sum left and right motor activity
    const scalar leftMotor = std::accumulate(&rCPU1[0], &rCPU1[8], 0.0f);
    const scalar rightMotor = std::accumulate(&rCPU1[8], &rCPU1[16], 0.0f);

    // Steer based on signal
    const scalar steering = leftMotor - rightMotor;
    if(displaySteering) {
        std::cout << "Steer:" << steering << std::endl;
    }
    if(steering > steerThreshold) {
        motor.tank(1.0f, -1.0f);
    }
    else if(steering < -steerThreshold) {
        motor.tank(-1.0f, 1.0f);
    }
    else {
        motor.tank(1.0f, 1.0f);
    }
}