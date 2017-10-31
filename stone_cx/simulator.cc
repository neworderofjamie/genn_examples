// Standard C++ includes
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Standard C includes
#include <cmath>
#include <cstdlib>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Common includes
#include "../common/analogue_csv_recorder.h"
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

template<typename F>
void drawNeuronActivity(scalar activity, const cv::Point &position, F getColourFn, cv::Mat &image)
{
    // Convert activity to a 8-bit level
    const unsigned char gray = (unsigned char)(255.0f * std::min(1.0f, std::max(0.0f, activity)));

    // Draw rectangle of this colour
    cv::rectangle(image, position, position + cv::Point(25, 25), getColourFn(gray), cv::FILLED);
}

template<typename F>
void drawPopulationActivity(scalar *popActivity, int popSize, const char *popName,
                            const cv::Point &position, F getColourFn, cv::Mat &image, int numColumns=0)
{
    // If (invalid) default number of columns is specified, use popsize
    if(numColumns == 0) {
        numColumns = popSize;
    }

    // Loop through each neuron in population
    for(int i = 0; i < popSize; i++) {
        // Calculate coordinate in terms of rows and columns
        auto coord = std::div(i, numColumns);
        cv::Point offset(coord.rem * 27, coord.quot * 27);

        // Draw neuron activity
        drawNeuronActivity(popActivity[i], position + offset, getColourFn, image);
    }

    // Label population
    const int numRows = (int)ceil((double)popSize / (double)numColumns);
    cv::putText(image, popName, position + cv::Point(0, 17 + (27 * numRows)),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0xFF, 0xFF, 0xFF));
}

cv::Scalar getReds(unsigned char gray)
{
    return CV_RGB(gray, 0, 0);
}

cv::Scalar getGreens(unsigned char gray)
{
    return CV_RGB(0, gray, 0);
}

cv::Scalar getBlues(unsigned char gray)
{
    return CV_RGB(0, 0, gray);
}
}   // Anonymous namespace

int main()
{
    const unsigned int pathImageSize = 1000;
    const unsigned int activityImageWidth = 500;
    const unsigned int activityImageHeight = 1000;
    const double pi = 3.141592653589793238462643383279502884;

    allocateMem();
    initialize();

    //---------------------------------------------------------------------------
    // Initialize neuron parameters
    //---------------------------------------------------------------------------
    // TN2
    preferredAngleTN2[Parameters::HemisphereLeft] = pi / 4.0;
    preferredAngleTN2[Parameters::HemisphereRight] = -pi / 4.0;

    // TL
    for(unsigned int i = 0; i < 8; i++) {
        preferredAngleTL[i] = preferredAngleTL[8 + i] = (pi / 4.0) * (double)i;
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

    initstone_cx();

    cv::namedWindow("Path", CV_WINDOW_NORMAL);
    cv::resizeWindow("Path", pathImageSize, pathImageSize);
    cv::Mat pathImage(pathImageSize, pathImageSize, CV_8UC3, cv::Scalar::all(0));

    cv::namedWindow("Activity", CV_WINDOW_NORMAL);
    cv::resizeWindow("Activity", activityImageWidth, activityImageHeight);
    cv::moveWindow("Activity", pathImageSize, 0);
    cv::Mat activityImage(activityImageHeight, activityImageWidth, CV_8UC3, cv::Scalar::all(0));

    // Create Von Mises distribution to sample angular acceleration from
    std::array<uint32_t, std::mt19937::state_size> seedData;
    std::random_device seedSource;
    std::generate(seedData.begin(), seedData.end(),
                  [&seedSource](){ return seedSource(); });
    std::seed_seq seeds(std::begin(seedData), std::end(seedData));
    std::mt19937 gen(seeds);

    VonMisesDistribution<double> pathVonMises(0.0, Parameters::pathKappa);

    // Create acceleration spline
    tk::spline accelerationSpline;
    {
        // Create vectors to hold the times at which linear acceleration
        // should change and it's values at those time
        const unsigned int numAccelerationChanges = Parameters::numOutwardTimesteps / 50;
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

#ifdef RECORD_ELECTROPHYS
    AnalogueCSVRecorder<scalar> tn2Recorder("tn2.csv", rTN2, Parameters::numTN2, "TN2");
    AnalogueCSVRecorder<scalar> cl1Recorder("cl1.csv", rCL1, Parameters::numCL1, "CL1");
    AnalogueCSVRecorder<scalar> tb1Recorder("tb1.csv", rTB1, Parameters::numTB1, "TB1");
    AnalogueCSVRecorder<scalar> cpu4Recorder("cpu4.csv", rCPU4, Parameters::numCPU4, "CPU4");
    AnalogueCSVRecorder<scalar> cpu1Recorder("cpu1.csv", rCPU1, Parameters::numCPU1, "CPU1");
#endif  // RECORD_ELECTROPHYS

    // Simulate
    double omega = 0.0;
    double theta = 0.0;
    double xVelocity = 0.0;
    double yVelocity = 0.0;
    double xPosition = 0.0;
    double yPosition = 0.0;
    for(unsigned int i = 0; i < (Parameters::numOutwardTimesteps + Parameters::numInwardTimesteps); i++) {
        // Update TN2 input input
        headingAngleTN2 = theta;
        vXTN2 = xVelocity;
        vYTN2 = yVelocity;

        // Update TL input
        headingAngleTL = theta;

        // Step network
        stepTimeCPU();

#ifdef RECORD_ELECTROPHYS
        tn2Recorder.record(i);
        cl1Recorder.record(i);
        tb1Recorder.record(i);
        cpu4Recorder.record(i);
        cpu1Recorder.record(i);
#endif  // RECORD_ELECTROPHYS

        // Draw compass system activity
        drawPopulationActivity(rTL, Parameters::numTL, "TL", cv::Point(10, 10),
                               getReds, activityImage, 8);
        drawPopulationActivity(rCL1, Parameters::numCL1, "CL1", cv::Point(10, 110),
                               getReds, activityImage, 8);
        drawPopulationActivity(rTB1, Parameters::numTB1, "TB1", cv::Point(10, 210),
                               getReds, activityImage);

        drawPopulationActivity(rTN2, Parameters::numTN2, "TN2", cv::Point(300, 310),
                               getBlues, activityImage, 1);

        drawPopulationActivity(rCPU4, Parameters::numCPU4, "CPU4", cv::Point(10, 310),
                               getGreens, activityImage, 8);
        drawPopulationActivity(rPontine, Parameters::numPontine, "Pontine", cv::Point(10, 410),
                               getGreens, activityImage, 8);
        drawPopulationActivity(rCPU1, Parameters::numCPU1, "CPU1", cv::Point(10, 510),
                               getGreens, activityImage, 8);

        // If we are on outbound segment of route
        const bool outbound = (i < Parameters::numOutwardTimesteps);
        double a = 0.0;
        if(outbound) {
            // Update angular velocity
            omega = (Parameters::pathLambda * omega) + pathVonMises(gen);

            // Read linear acceleration off spline
            a = accelerationSpline((double)i);
        }
        // Otherwise we're path integrating home
        else {
            // Sum left and right motor activity
            const scalar leftMotor = std::accumulate(&rCPU1[0], &rCPU1[8], 0.0f);
            const scalar rightMotor = std::accumulate(&rCPU1[8], &rCPU1[16], 0.0f);

            // Use difference between left and right to calculate angular velocity
            omega = -Parameters::agentM * (rightMotor - leftMotor);

            // Use fixed acceleration
            a = 0.1;
        }

        // Update heading
        theta += omega;

        // Update linear velocity
        // **NOTE** this comes from https://github.com/InsectRobotics/path-integration/blob/master/bee_simulator.py#L77-L83 rather than the methods section
        xVelocity += sin(theta) * a;
        yVelocity += cos(theta) * a;
        xVelocity -= Parameters::agentDrag * xVelocity;
        yVelocity -= Parameters::agentDrag * yVelocity;

        // Update position
        xPosition += xVelocity;
        yPosition += yVelocity;

        // Draw agent position (centring so origin is in centre of path image)
        const cv::Point p((pathImageSize / 2) + (int)xPosition, (pathImageSize / 2) + (int)yPosition);
        cv::line(pathImage, p, p,
                 outbound ? CV_RGB(0xFF, 0, 0) : CV_RGB(0, 0xFF, 0));

        // Show output image
        cv::imshow("Path", pathImage);
        cv::imshow("Activity", activityImage);
        cv::waitKey(1);
    }
    return 0;
}