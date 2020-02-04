// Standard C includes
#include <cassert>
#include <cmath>

//#define USE_ZERO_COPY
//#define JETSON_POWER


// Parameters
namespace Parameters
{
// Layers of model
enum Layer
{
    Layer23,
    Layer4,
    Layer5,
    Layer6,
    LayerMax,
};

// Populations in each layer of model
enum Population
{
    PopulationE,
    PopulationI,
    PopulationMax,
};

// Layer names
const char *layerNames[LayerMax] = {
    "23",
    "4",
    "5",
    "6"
};

// Population names
const char *populationNames[PopulationMax] = {
    "E",
    "I",
};

// Should we measure timing?
const bool measureTiming = true;

// Should we use pre or postsynaptic parallelism?
const bool presynapticParallelism = false;

// Should we use procedural rather than in-memory connectivity?
const bool proceduralConnectivity = false;

// Assert settings are valid
static_assert(presynapticParallelism || !proceduralConnectivity,
              "Procedural connectivity can only be use with presynaptic parallelism");

// Number of threads to use for each row if using presynaptic parallelism
const unsigned int numThreadsPerSpike = 1;

// Simulation timestep [ms]
const double dtMs = 0.1;

// Simulation duration [ms]
const double durationMs = 1000.0;

// Scaling factors for number of neurons and synapses
const double neuronScalingFactor = 1.0;
const double connectivityScalingFactor = 1.0;

// Background rate per synapse
const double backgroundRate = 8.0;  // spikes/s

// Relative inhibitory synaptic weight
const double g = -4.0;

// Mean synaptic weight for all excitatory projections except L4e->L2/3e
const double meanW = 87.8e-3;  // nA
const double externalW = 87.8e-3;   // nA

// Mean synaptic weight for L4e->L2/3e connections
// See p. 801 of the paper, second paragraph under 'Model Parameterization',
// and the caption to Supplementary Fig. 7
const double layer234W = 2.0 * meanW;   // nA

// Standard deviation of weight distribution relative to mean for
// all projections except L4e->L2/3e
const double relW = 0.1;

// Standard deviation of weight distribution relative to mean for L4e->L2/3e
// This value is not mentioned in the paper, but is chosen to match the
// original code by Tobias Potjans
const double layer234RelW = 0.05;

// Numbers of neurons in full-scale model
//  PopulationE,    PopulationI
const unsigned int numNeurons[LayerMax][PopulationMax] = {
    {20683,         5834},  // Layer23
    {21915,         5479},  // Layer4
    {4850,          1065},   // Layer5
    {14395,         2948}};  // Layer6

// Probabilities for >=1 connection between neurons in the given populations.
// The first index is for the target population; the second for the source population
//  2/3e        2/3i    4e      4i      5e      5i      6e      6i
const double connectionProbabilities[LayerMax * PopulationMax][LayerMax * PopulationMax] = {
    {0.1009,    0.1689, 0.0437, 0.0818, 0.0323, 0.0,    0.0076, 0.0},       // 2/3e
    {0.1346,    0.1371, 0.0316, 0.0515, 0.0755, 0.0,    0.0042, 0.0},       // 2/3i
    {0.0077,    0.0059, 0.0497, 0.135,  0.0067, 0.0003, 0.0453, 0.0},       // 4e
    {0.0691,    0.0029, 0.0794, 0.1597, 0.0033, 0.0,    0.1057, 0.0},       // 4i
    {0.1004,    0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0},       // 5e
    {0.0548,    0.0269, 0.0257, 0.0022, 0.06,   0.3158, 0.0086, 0.0},       // 5i
    {0.0156,    0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252},    // 6e
    {0.0364,    0.001,  0.0034, 0.0005, 0.0277, 0.008,  0.0658, 0.1443}};   // 6i

// In-degrees for external inputs
//  PopulationE,    PopulationI
const unsigned int numExternalInputs[LayerMax][PopulationMax] = {
    {1600,          1500},  // Layer23
    {2100,          1900},  // Layer4
    {2000,          1900},  // Layer5
    {2900,          2100}}; // Layer6

// Mean rates in the full-scale model, necessary for scaling
// Precise values differ somewhat between network realizations
//  PopulationE,    PopulationI
const double meanFiringRates[LayerMax][PopulationMax] = {
    {0.971,         2.868},     // Layer23
    {4.746,         5.396},     // Layer4
    {8.142,         9.078},     // Layer5
    {0.991,         7.523}};    // Layer6


// Means and standard deviations of delays from given source populations (ms)
const double meanDelay[PopulationMax] = {
    1.5,    // PopulationE
    0.75};  // PopulationI

const double delaySD[PopulationMax] = {
    0.75,   // PopulationE
    0.375}; // PopulationI


std::string getPopulationName(unsigned int layer, unsigned int population)
{
    return std::string(layerNames[layer]) + std::string(populationNames[population]);
}

unsigned int getScaledNumNeurons(unsigned int layer, unsigned int pop)
{
    return (unsigned int)(neuronScalingFactor * (double)numNeurons[layer][pop]);
}

double getFullNumInputs(unsigned int srcLayer, unsigned int srcPop, unsigned int trgLayer, unsigned int trgPop)
{
    const unsigned numSrc = numNeurons[srcLayer][srcPop];
    const unsigned numTrg = numNeurons[trgLayer][trgPop];
    const double connectionProb = connectionProbabilities[(trgLayer * PopulationMax) + trgPop][(srcLayer * PopulationMax) + srcPop];

    return round(log(1.0 - connectionProb) / log((double)(numTrg * numSrc - 1) / (double)(numTrg * numSrc))) / numTrg;
}

double getMeanWeight(unsigned int srcLayer, unsigned int srcPop, unsigned int trgLayer, unsigned int trgPop)
{
    // Determine mean weight
    if(srcPop == PopulationE) {
        if(srcLayer == Layer4 && trgLayer == Layer23 && trgPop == PopulationE) {
            return layer234W;
        }
        else {
            return Parameters::meanW;
        }
    }
    else {
        return g * meanW;
    }
}

unsigned int getScaledNumConnections(unsigned int srcLayer, unsigned int srcPop, unsigned int trgLayer, unsigned int trgPop)
{
    // Scale full number of inputs by scaling factor
    const double numInputs = getFullNumInputs(srcLayer, srcPop, trgLayer, trgPop) * connectivityScalingFactor;
    assert(numInputs >= 0.0);

    // Multiply this by number of postsynaptic neurons
    return (unsigned int)(round(numInputs * (double)getScaledNumNeurons(trgLayer, trgPop)));

}

double getFullMeanInputCurrent(unsigned int layer, unsigned int pop)
{
    // Loop through source populations
    double meanInputCurrent = 0.0;
    for(unsigned int srcLayer = 0; srcLayer < LayerMax; srcLayer++) {
        for(unsigned int srcPop = 0; srcPop < PopulationMax; srcPop++) {
            meanInputCurrent += (getMeanWeight(srcLayer, srcPop, layer, pop) *
                                 getFullNumInputs(srcLayer, srcPop, layer, pop) *
                                 meanFiringRates[srcLayer][srcPop]);
        }
    }

    // Add mean external input current
    meanInputCurrent += externalW * numExternalInputs[layer][pop] * backgroundRate;
    assert(meanInputCurrent >= 0.0);
    return meanInputCurrent;
}
}   // Parameters
