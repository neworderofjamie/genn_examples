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
#include "../common/von_mises_distribution.h"

// GeNN generated code includes
#include "stone_cx_CODE/definitions.h"

// Model includes
#include "parameters.h"
#include "simulatorCommon.h"
#include "spline.h"

//---------------------------------------------------------------------------
// Anonymous namespace
//---------------------------------------------------------------------------
namespace
{

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
    buildConnectivity();

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