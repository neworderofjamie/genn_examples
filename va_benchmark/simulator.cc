#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

#include "../common/connectors.h"
#include "../common/spike_csv_recorder.h"

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

  std::random_device rd;
  std::mt19937 gen(rd());

  buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                 CII, &allocateII, gen);
  buildFixedProbabilityConnector(Parameters::numInhibitory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                 CIE, &allocateIE, gen);
  buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numExcitatory, Parameters::probabilityConnection,
                                 CEE, &allocateEE, gen);
  buildFixedProbabilityConnector(Parameters::numExcitatory, Parameters::numInhibitory, Parameters::probabilityConnection,
                                 CEI, &allocateEI, gen);

  // Final setup
  initva_benchmark();

  // Randomlise initial membrane voltages
  std::uniform_real_distribution<> dis(Parameters::resetVoltage, Parameters::thresholdVoltage);
  for(unsigned int i = 0; i < Parameters::numExcitatory; i++)
  {
    VE[i] = dis(gen);
  }

  for(unsigned int i = 0; i < Parameters::numInhibitory; i++)
  {
    VI[i] = dis(gen);
  }

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
