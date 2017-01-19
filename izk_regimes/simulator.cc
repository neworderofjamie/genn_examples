#include "model.cc"
#include "izk_regimes_CODE/definitions.h"

int main()
{
  allocateMem();
  initialize();

  /*labels = ["Regular spiking", "Fast spiking", "Chattering", "Intrinsically bursting"]
  a = [0.02,  0.1,    0.02,   0.02]
  b = [0.2,   0.2,    0.2,    0.2]
  c = [-65.0, -65.0,  -50.0,  -55.0]
  d = [8.0,   2.0,    2.0,    4.0]*/
  // Regular
  aNeurons[0] = 0.02;
  bNeurons[0] = 0.2;
  cNeurons[0] = -65.0;
  dNeurons[0] = 8.0;

  // Fast
  aNeurons[1] = 0.1;
  bNeurons[1] = 0.2;
  cNeurons[1] = -65.0;
  dNeurons[1] = 2.0;

  // Chattering
  aNeurons[2] = 0.02;
  bNeurons[2] = 0.2;
  cNeurons[2] = -50.0;
  dNeurons[2] = 2.0;

  // Bursting
  aNeurons[3] = 0.02;
  bNeurons[3] = 0.2;
  cNeurons[3] = -55.0;
  dNeurons[3] = 4.0;


  FILE *membraneVoltage = fopen("voltages.csv", "w");
  fprintf(membraneVoltage, "Time(ms), Neuron number, Voltage (mV)\n");

  // Advance time
  for(double t = 0.0; t < 200.0; t += 0.1)
  {
    // Run timestep
    stepTimeCPU();

    for(unsigned int n = 0; n < 4; n++)
    {
      fprintf(membraneVoltage, "%f, %u, %f\n", t, n, VNeurons[n]);
    }
  }

  fclose(membraneVoltage);

  return 0;
}