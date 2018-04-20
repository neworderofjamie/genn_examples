#pragma once

namespace Parameters
{
    // Topology of model
    const unsigned int gridSize = 2;
    const unsigned int numHC = gridSize * gridSize;

    // Numbers of neurons in each hypercolumn (HC)
    const unsigned int numHCExcitatoryNeurons = 1000;
    const unsigned int numHCInhibitoryNeurons = 250;

    // Numbers of neurosn in each minicolumn (MC)
    const unsigned int numMCNeurons = 100;

    // Connection probability
    const double probabilityConnection = 0.1;

    // Number of excitatory synapses on neuron
    const unsigned int numESynapses = (unsigned int)std::round(probabilityConnection * (double)numHCExcitatoryNeurons);

    // Neuron parameters
    const double neuronResetVoltage = -70.0;
    const double neuronThresholdVoltage = -50.0;
    const double neuronTauMem = 20.0;

    // Synaptic time constant
    const double tauSynAMPAGABA = 5.0;
    const double tauSynNMDA = 150.0;

    // BCPNN time constants
    const double tauZiAMPA = tauSynAMPAGABA;
    const double tauZjAMPA = tauSynAMPAGABA;
    const double tauZiNMDA = tauSynNMDA;
    const double tauZjNMDA = tauSynAMPAGABA;
    const double tauP = 2000.0;

    // Maximum firing frequency
    const double fmax = 20.0;

    // Weights for static synapses
    const double effectiveWeight = 25.0;
    const double staticExcitatoryWeight = (effectiveWeight / tauSynAMPAGABA) * 0.00041363506632638 * 250.0;
    const double staticInhibitoryWeight = -5.0 * staticExcitatoryWeight;

    // Background input weight for all neurons
    const double backgroundWeightTraining = 0.2;

    // Stimuli input weight for all neurons
    const double stimWeightTraining = 2.0;

    // Background input rate for all neurons
    const double backgroundRate = 1.3 * 1000.0 * ((neuronThresholdVoltage - neuronResetVoltage) / (effectiveWeight * neuronTauMem));
}