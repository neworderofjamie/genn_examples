#include <algorithm>
#include<random>

#include "model.cc"
#include "vogels_2011_CODE/definitions.h"

typedef void (*allocateFn)(unsigned int);

template<typename Generator>
void build_fixed_probability_connector(unsigned int numPre, unsigned int numPost, float probability,
                                       SparseProjection &projection, allocateFn allocate, Generator &gen)
{
  // Allocate memory for indices
  // **NOTE** RESIZE as this vector is populated by index
  std::vector<unsigned int> tempIndInG;
  tempIndInG.resize(numPre + 1);

  // Reserve a temporary vector to store indices
  std::vector<unsigned int> tempInd;
  tempInd.reserve((unsigned int)((float)(numPre * numPost) * probability));

  // Create RNG to draw probabilities
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // Loop through pre neurons
  for(unsigned int i = 0; i < numPre; i++)
  {
    // Connections from this neuron start at current end of indices
    tempIndInG[i] = tempInd.size();

    // Loop through post neurons
    for(unsigned int j = 0; j < numPost; j++)
    {
      // If there should be a connection here, add one to temporary array
      if(dis(gen) < probability)
      {
        tempInd.push_back(j);
      }
    }
  }

  // Add final index
  tempIndInG[numPre] = tempInd.size();

  // Allocate SparseProjection arrays
  // **NOTE** shouldn't do directly as underneath it may use CUDA or host functions
  allocate(tempInd.size());

  // Copy indices
  std::copy(tempIndInG.begin(), tempIndInG.end(), &projection.indInG[0]);
  std::copy(tempInd.begin(), tempInd.end(), &projection.ind[0]);
}

int main()
{
  allocateMem();

  initialize();

  std::random_device rd;
  std::mt19937 gen(rd());

  build_fixed_probability_connector(500, 500, 0.02f,
                                    CII, &allocateII, gen);
  build_fixed_probability_connector(500, 2000, 0.02f,
                                    CIE, &allocateIE, gen);
  build_fixed_probability_connector(2000, 2000, 0.02f,
                                    CEE, &allocateEE, gen);
  build_fixed_probability_connector(2000, 500, 0.02f,
                                    CEI, &allocateEI, gen);

  // Copy conductances
  std::fill(&gII[0], &gII[CII.connN], -0.03);
  std::fill(&gIE[0], &gIE[CIE.connN], 0.0);
  std::fill(&gEE[0], &gEE[CEE.connN], 0.03);
  std::fill(&gEI[0], &gEI[CEI.connN], 0.03);

  // Setup reverse connection indices for STDP
  initvogels_2011();

  // Randomlise initial membrane voltages
  std::uniform_real_distribution<> dis(-60.0, -50.0);
  for(unsigned int i = 0; i < 2000; i++)
  {
    VE[i] = dis(gen);
  }

  for(unsigned int i = 0; i < 500; i++)
  {
    VI[i] = dis(gen);
  }

  // Open CSV output files
  FILE *spikes = fopen("spikes.csv", "w");
  fprintf(spikes, "Time(ms), Neuron ID\n");
  FILE *weights = fopen("weights.csv", "w");
  fprintf(weights, "Time(ms), Weight (nA)\n");

  // Loop through timesteps
  for(unsigned int t = 0; t < 10000; t++)
  {
    // Simulate
    stepTimeCPU();

    // Write spike times to file
    for(unsigned int i = 0; i < glbSpkCntE[0]; i++)
    {
      fprintf(spikes, "%f, %u\n", 1.0 * (double)t, glbSpkE[i]);
    }

    // Calculate mean IE weights
    float totalWeight = std::accumulate(&gIE[0], &gIE[CIE.connN], 0.0f);
    fprintf(weights, "%f, %f\n", 1.0 * (double)t, totalWeight / (double)CIE.connN);

  }

  // Close files
  fclose(spikes);
  fclose(weights);

  return 0;
}
