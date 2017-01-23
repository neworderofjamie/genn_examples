#include <algorithm>
#include<random>

#include "model.cc"
#include "vogels_2011_CODE/definitions.h"

template<typename Generator>
void build_fixed_probability_connector(unsigned int numPre, unsigned int numPost, float probability,
                                       SparseProjection &projection, bool plastic, Generator &gen)
{
  // Allocate memory for indices
  projection.indInG = new unsigned int[numPre + 1];

  // Reserve a temporary vector to store indices
  std::vector<unsigned int> tempInd;
  tempInd.reserve((unsigned int)((float)(numPre * numPost) * probability));

  // Create RNG to draw probabilities
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // Loop through pre neurons
  for(unsigned int i = 0; i < numPre; i++)
  {
    // Connections from this neuron start at current end of indices
    projection.indInG[i] = tempInd.size();

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

  projection.connN = tempInd.size();
  projection.indInG[numPre] = projection.connN;


  // Allocate memory for indices and copy in temporary indices
  projection.ind = new unsigned int[projection.connN];
  std::copy(tempInd.begin(), tempInd.end(), &projection.ind[0]);

  // If connection is plastic, allocate extra data
  // structures to allow lookup by postsynaptic index
  if(plastic)
  {
    projection.revIndInG = new unsigned int[numPost + 1];
    projection.revInd= new unsigned int[projection.connN];
    projection.remap= new unsigned int[projection.connN];
  }
}

int main()
{
  allocateMem();

  initialize();

  std::random_device rd;
  std::mt19937 gen(rd());

  build_fixed_probability_connector(500, 500, 0.02f,
                                    CII, false, gen);
  build_fixed_probability_connector(500, 2000, 0.02f,
                                    CIE, true, gen);
  build_fixed_probability_connector(2000, 2000, 0.02f,
                                    CEE, false, gen);
  build_fixed_probability_connector(2000, 500, 0.02f,
                                    CEI, false, gen);

  // Allocate synapse conductances
  gII = new float[CII.connN];
  gIE = new float[CIE.connN];
  gEE = new float[CEE.connN];
  gEI = new float[CEI.connN];
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