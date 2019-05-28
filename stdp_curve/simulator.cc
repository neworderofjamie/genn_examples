#include "stdp_curve_CODE/definitions.h"

#define NUM_NEURONS 14

int main()
{
  allocateMem();

  initialize();

  // Setup reverse connection indices for STDP
  initializeSparse();

  // Spike pair configuration
  const double startTime = 200.0;
  const double timeBetweenPairs = 1000.0;
  const double deltaT[NUM_NEURONS] = {-100.0, -60.0, -40.0, -30.0, -20.0, -10.0, -1.0,
    1.0, 10.0, 20.0, 30.0, 40.0, 60.0, 100.0};

  // Loop through neurons
  unsigned int preSpikeTimesteps[NUM_NEURONS][60];
  unsigned int postSpikeTimesteps[NUM_NEURONS][60];
  unsigned int nextPreSpikeIndex[NUM_NEURONS];
  unsigned int nextPostSpikeIndex[NUM_NEURONS];
  for(unsigned int n = 0; n < NUM_NEURONS; n++)
  {
    // Start each spike source at first spike
    nextPreSpikeIndex[n] = 0;
    nextPostSpikeIndex[n] = 0;

    // Calculate spike timings
    const double neuronDeltaT = deltaT[n];
    const double prePhase = (neuronDeltaT > 0) ? (startTime + neuronDeltaT + 1.0) : (startTime + 1.0);
    const double postPhase = (neuronDeltaT > 0) ? startTime : (startTime - neuronDeltaT);

    printf("Neuron %u(%f): pre phase %f, post phase %f\n", n, neuronDeltaT, prePhase, postPhase);
    // Fill in spike timings
    for(unsigned int p = 0; p < 60; p++)
    {
      preSpikeTimesteps[n][p] = prePhase + ((double)p * timeBetweenPairs);
      postSpikeTimesteps[n][p] = postPhase + ((double)p * timeBetweenPairs);
    }
  }

  FILE *spikes = fopen("spikes.csv", "w");
  fprintf(spikes, "Time(ms), Neuron ID\n");

  // Loop through timesteps
  for(unsigned int t = 0; t < 60200; t++)
  {
    // Loop through spike sources
    for(unsigned int n = 0; n < NUM_NEURONS; n++)
    {
      // If there are more pre-spikes to emit and
      // the next one should be emitted this timestep
      if(nextPreSpikeIndex[n] < 60
        && preSpikeTimesteps[n][nextPreSpikeIndex[n]] == t)
      {
        // Manually add a spike to spike source's output
        glbSpkPreStim[glbSpkCntPreStim[0]++] = n;

        // **YUCK** also update the time used for post-after-pre STDP calculations
        sTPreStim[n] = 1.0 * (double)t;

        // Go onto next pre-spike
        nextPreSpikeIndex[n]++;
      }

      // If there are more post-spikes to emit and
      // the next one should be emitted this timestep
      if(nextPostSpikeIndex[n] < 60
        && postSpikeTimesteps[n][nextPostSpikeIndex[n]] == t)
      {
        // Manually add a spike to spike source's output
        glbSpkPostStim[glbSpkCntPostStim[0]++] = n;

        // Go onto next post-spike
        nextPostSpikeIndex[n]++;
      }
    }
    // Simulate
    stepTime();

    // Write spike times to file
    for(unsigned int i = 0; i < glbSpkCntExcitatory[0]; i++) {
      fprintf(spikes, "%f, %u\n", 1.0 * (double)t, glbSpkExcitatory[i]);
    }
  }
  fclose(spikes);

  FILE *weights = fopen("weights.csv", "w");
  fprintf(weights, "Delta T [ms], Weight\n");

  for(unsigned int n = 0; n < NUM_NEURONS; n++)
  {
    fprintf(weights, "%f, %f\n", deltaT[n], gPreStimToExcitatory[n]);
  }

  fclose(weights);


  return 0;
}
