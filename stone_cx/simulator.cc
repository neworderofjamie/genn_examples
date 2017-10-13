// Standard C++ includes
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Standard C includes
#include <cmath>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Common includes
#include "../common/connectors.h"
#include "../common/von_mises_distribution.h"

// GeNN generated code includes
#include "stone_cx_CODE/definitions.h"

// Model includes
#include "parameters.h"
#include "spline.h"

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

int main()
{
    const unsigned int worldSize = 1000;
    const double pi = 3.141592653589793238462643383279502884;

    allocateMem();
    initialize();

    //---------------------------------------------------------------------------
    // Initialize neuron parameters
    //---------------------------------------------------------------------------
    // TL
    preferredAngleTL[Parameters::HemisphereLeft] = pi / 4.0;
    preferredAngleTL[Parameters::HemisphereRight] = -pi / 4.0;

    //---------------------------------------------------------------------------
    // TN2
    //---------------------------------------------------------------------------
    for(unsigned int i = 0; i < 8; i++) {
        preferredAngleTN2[i] = preferredAngleTN2[8 + i] = (pi / 4.0) * (double)i;
    }

    //---------------------------------------------------------------------------
    // Build connectivity
    //---------------------------------------------------------------------------
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
            const double preferredI = preferredAngleTN2[i];
            const double preferredJ = preferredAngleTN2[j];

            const double w = (cos(preferredI - preferredJ) - 1.0) / 2.0;
            gTB1_TB1[(i * Parameters::numTB1) + j] = Parameters::c * w;
        }
    }
    //std::cout << "TB1->TB1" << std::endl;
    //printSparseMatrix(Parameters::numTL, CTL_CL1);

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
        CCPU4_CPU1.ind[1 + i] = 7 + i;
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

    initstone_cx();

    cv::namedWindow("Path", CV_WINDOW_NORMAL);
    cv::resizeWindow("Path", worldSize, worldSize);

    // Create output image
    cv::Mat outputImage(worldSize, worldSize, CV_8UC3, cv::Scalar::all(0));

    // Create Von Mises distribution to sample angular acceleration from
    std::mt19937 gen;
    VonMisesDistribution<double> pathVonMises(0.0, Parameters::pathKappa);

    // Create acceleration spline
    tk::spline accelerationSpline;
    {
        // Create vectors to hold the times at which linear acceleration
        // should change and it's values at those time
        const unsigned int numAccelerationChanges = Parameters::numTimesteps / 50;
        std::vector<double> accelerationTime(numAccelerationChanges);
        std::vector<double> accelerationMagnitude(numAccelerationChanges);

        // Draw accelerations from real distribution
        std::uniform_real_distribution<double> acceleration(Parameters::agentMinAcceleration,
                                                            Parameters::agentMaxAcceleration);
        std::generate(accelerationMagnitude.begin(), accelerationMagnitude.end(),
                      [&gen, &acceleration](){ return acceleration(gen); });

        for(unsigned int i = 0; i < numAccelerationChanges; i++) {
            accelerationTime[i] = i * 50;
        }

        // Build spline from these
        accelerationSpline.set_points(accelerationTime, accelerationMagnitude);
    }

    double omega = 0.0;
    double theta = 0.0;
    double xVelocity = 0.0;
    double yVelocity = 0.0;
    double xPosition = 0.0;
    double yPosition = 0.0;
    for(unsigned int i = 0; i < Parameters::numTimesteps; i++) {
        // Update angular velocity and thus heading of agent
        omega = (Parameters::pathLambda * omega) + pathVonMises(gen);
        theta += omega;

        // Read linear acceleration off spline
        const double a = accelerationSpline((double)i);

        // Update linear velocity
        // **NOTE** this comes from https://github.com/InsectRobotics/path-integration/blob/master/bee_simulator.py#L77-L83 rather than paper
        xVelocity += sin(theta) * a;
        yVelocity += cos(theta) * a;
        xVelocity -= Parameters::agentDrag * xVelocity;
        yVelocity -= Parameters::agentDrag * yVelocity;

        // Update position
        xPosition += xVelocity;
        yPosition += yVelocity;

        // Draw agent position
        const cv::Point p(500 + (int)xPosition, 500 + (int)yPosition);
        cv::line(outputImage, p, p, CV_RGB(0xFF, 0xFF, 0xFF));

        // Show output image
        cv::imshow("Path", outputImage);
        cv::waitKey(33);
    }
    return 0;
}