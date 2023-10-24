#include <random>

void modelDefinition(ModelSpec &model)
{
    GENN_PREFERENCES.debugCode = true;
    model.setDT(1.0);
    model.setName("spike_source_array");

    VarValues ssaInit{
        {"startSpike", uninitialisedVar()},
        {"endSpike", uninitialisedVar()}};

    auto *n = model.addNeuronPopulation<NeuronModels::SpikeSourceArray>(
        "SSA", 100, {}, ssaInit);
    n->setSpikeRecordingEnabled(true);
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
    const float rateHz = 10.0f;
    const float durationMs = 500.0f;

    const float scale = 1000.0f / (rateHz * model.getDT());
    std::mt19937 rng;
    std::exponential_distribution<float> dist;
    std::vector<float> spikeTimes;
    std::vector<unsigned int> endIndices;
    endIndices.reserve(101);

    endIndices.push_back(0);
    // Loop through neurons
    for(size_t n = 0; n < 100; n++) {
        // Generate poisson spike train
        float time = 0.0f;
        while(true) {
            time += scale * dist(rng);
            if(time >= durationMs) {
                break;
            }
            else {
                spikeTimes.push_back(time);
            }

        }

        // Add end index
        endIndices.push_back((unsigned int)spikeTimes.size());
    }

    std::cout << spikeTimes.size() << std::endl;
    runtime.allocate(500);
    runtime.initialize();

    const auto *n = model.findNeuronGroup("SSA");
    std::copy_n(endIndices.cbegin(), 100, runtime.getArray(*n, "startSpike")->getHostPointer<unsigned int>());
    std::copy_n(endIndices.cbegin() + 1, 100, runtime.getArray(*n, "endSpike")->getHostPointer<unsigned int>());

    runtime.initializeSparse();

    // Allocate spike times EGP, copy in data and push
    runtime.allocateArray(*n, "spikeTimes", spikeTimes.size());
    std::copy(spikeTimes.cbegin(), spikeTimes.cend(), runtime.getArray(*n, "spikeTimes")->getHostPointer<float>());
    runtime.getArray(*n, "spikeTimes")->pushToDevice();

    // Simulate
    while(runtime.getTime() < durationMs) {
        runtime.stepTime();
    }

    // Pull spikes and write to CSV
    runtime.pullRecordingBuffersFromDevice();
    runtime.writeRecordedSpikes(*n, "spikes.csv");
}
