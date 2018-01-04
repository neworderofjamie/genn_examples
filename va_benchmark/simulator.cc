#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

// GeNN robotics includes
#include "connectors.h"
#include "spike_csv_recorder.h"

#include "parameters.h"

#include "va_benchmark_CODE/definitions.h"

int main()
{
  auto  allocStart = chrono::steady_clock::now();
  allocateMem();
  auto  allocEnd = chrono::steady_clock::now();
  printf("Allocation %ldms\n", chrono::duration_cast<chrono::milliseconds>(allocEnd - allocStart).count());

  auto  initStart = chrono::steady_clock::now(); 
  initialize();

#ifndef CPU_ONLY
  std::mt19937 rng;
#endif
  buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                 CII, &allocateII, rng);
  buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                 CIE, &allocateIE, rng);
  buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                 CEE, &allocateEE, rng);
  buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                 CEI, &allocateEI, rng);

  // Final setup
  initva_benchmark();

  auto  initEnd = chrono::steady_clock::now();
  printf("Init %ldms\n", chrono::duration_cast<chrono::milliseconds>(initEnd - initStart).count());

  // Open CSV output files
  SpikeCSVRecorder spikes("spikes.csv", glbSpkCntE, glbSpkE);

  auto simStart = chrono::steady_clock::now();
  // Loop through timesteps
  for(unsigned int t = 0; t < 10000; t++)
  {
    // Simulate
#ifndef CPU_ONLY
    stepTimeGPU();

    pullECurrentSpikesFromDevice();
#else
    stepTimeCPU();
#endif

    spikes.record(t);
  }
  auto simEnd = chrono::steady_clock::now();
  printf("Simulation %ldms\n", chrono::duration_cast<chrono::milliseconds>(simEnd - simStart).count());

  return 0;
}
