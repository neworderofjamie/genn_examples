#pragma once


//------------------------------------------------------------------------
// SimParameters
//------------------------------------------------------------------------
namespace SimParameters
{
    // Outbound path generation parameters
    const unsigned int numOutwardTimesteps = 1500;
    const unsigned int numInwardTimesteps = 1500;

    const double pathLambda = 0.4;
    const double pathKappa = 100.0;

    const double agentDrag = 0.15;

    const double agentMinAcceleration = 0.0;
    const double agentMaxAcceleration = 0.15;
    const double agentM = 0.5;
}