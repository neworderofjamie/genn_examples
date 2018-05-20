#include "model.cc"
#include "adexp_curr_CODE/definitions.h"

int main()
{
  allocateMem();
  initialize();

  FILE *membraneVoltage = fopen("voltages.csv", "w");
  fprintf(membraneVoltage, "Time(ms), Voltage (mV)\n");

  // Loop through timesteps
  for(unsigned int i = 0; i < 2000; i++)
  {
    // Simulate
    stepTimeCPU();

    // Calculate simulation time
    const double time = 1.0 * (double)t;
    fprintf(membraneVoltage, "%f, %f, %f\n", time, VNeurons[0], WNeurons[0]);

  }

  fclose(membraneVoltage);

  return 0;
}