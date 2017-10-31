#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

#include "../common/connectors.h"
#include "../common/spike_csv_recorder.h"

#include "vogels_2011_CODE/definitions.h"

int main()
{
  auto  allocStart = chrono::steady_clock::now();
  allocateMem();
  auto  allocEnd = chrono::steady_clock::now();
  printf("Allocation %ldms\n", chrono::duration_cast<chrono::milliseconds>(allocEnd - allocStart).count());

  auto  initStart = chrono::steady_clock::now(); 
  initialize();

  buildFixedProbabilityConnector(500, 500, 0.02f,
                                 CII, &allocateII, rng);
  buildFixedProbabilityConnector(500, 2000, 0.02f,
                                 CIE, &allocateIE, rng);
  buildFixedProbabilityConnector(2000, 2000, 0.02f,
                                 CEE, &allocateEE, rng);
  buildFixedProbabilityConnector(2000, 500, 0.02f,
                                 CEI, &allocateEI, rng);

  // Setup reverse connection indices for STDP
  initvogels_2011();

  auto  initEnd = chrono::steady_clock::now();
  printf("Init %ldms\n", chrono::duration_cast<chrono::milliseconds>(initEnd - initStart).count());

  // Open CSV output files
  SpikeCSVRecorder spikes("spikes.csv", glbSpkCntE, glbSpkE);

  FILE *weights = fopen("weights.csv", "w");
  fprintf(weights, "Time(ms), Weight (nA)\n");
  auto simStart = chrono::steady_clock::now();
  // Loop through timesteps
  for(unsigned int t = 0; t < 10000; t++)
  {
    // Simulate
#ifndef CPU_ONLY
    stepTimeGPU();

    pullECurrentSpikesFromDevice();
    //pullIEStateFromDevice();
#else
    stepTimeCPU();
#endif

    spikes.record(t);


    // Calculate mean IE weights
    float totalWeight = std::accumulate(&gIE[0], &gIE[CIE.connN], 0.0f);
    fprintf(weights, "%f, %f\n", 1.0 * (double)t, totalWeight / (double)CIE.connN);

  }
  auto simEnd = chrono::steady_clock::now();
  printf("Simulation %ldms\n", chrono::duration_cast<chrono::milliseconds>(simEnd - simStart).count());

  // Close files
  fclose(weights);

  return 0;
}
