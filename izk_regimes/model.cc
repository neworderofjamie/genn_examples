#include <fstream>

//----------------------------------------------------------------------------
// IzhikevichVS
//----------------------------------------------------------------------------
class IzhikevichV : public NeuronModels::Izhikevich
{
public:
    DECLARE_SNIPPET(IzhikevichV);

    SET_SIM_CODE(
        "if (V >= 30.0){\n"
        "    V=c;\n"
        "    U+=d;\n"
        "} \n"
        "V+=0.5*(0.04*V*V+5.0*V+140.0-U+Isyn+Ioffset)*dt; //at two times for numerical stability\n"
        "V+=0.5*(0.04*V*V+5.0*V+140.0-U+Isyn+Ioffset)*dt;\n"
        "U+=a*(b*V-U)*dt;\n"
        "//if (V > 30.0){   //keep this only for visualisation -- not really necessaary otherwise \n"
        "//  V=30.0; \n"
        "//}\n"
    );
    
    SET_PARAMS({"Ioffset"});

    SET_VARS({{"V","scalar"}, {"U", "scalar"}, {"a", "scalar"}, {"b", "scalar"}, {"c", "scalar"}, {"d", "scalar"}});
};
IMPLEMENT_SNIPPET(IzhikevichV);

void modelDefinition(ModelSpec &model)
{
    model.setDT(0.1);
    model.setName("izk_regimes");

    // Izhikevich model parameters
    ParamValues paramValues{{"Ioffset", 10.0}};
    VarValues initValues{
        {"V", -65.0},
        {"U", -20.0},
        {"a", uninitialisedVar()},
        {"b", uninitialisedVar()},
        {"c", uninitialisedVar()},
        {"d", uninitialisedVar()}};

    // Create population of Izhikevich neurons
    model.addNeuronPopulation<IzhikevichV>("Neurons", 4, paramValues, initValues);
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
    const auto *neurons = model.findNeuronGroup("Neurons");
    
    runtime.allocate();
    runtime.initialize();
    float *aNeurons = runtime.getArray(*neurons, "a")->getHostPointer<float>();
    float *bNeurons = runtime.getArray(*neurons, "b")->getHostPointer<float>();
    float *cNeurons = runtime.getArray(*neurons, "c")->getHostPointer<float>();
    float *dNeurons = runtime.getArray(*neurons, "d")->getHostPointer<float>();
    
    // Regular
    aNeurons[0] = 0.02f;
    bNeurons[0] = 0.2f;
    cNeurons[0] = -65.0f;
    dNeurons[0] = 8.0f;

    // Fast
    aNeurons[1] = 0.1f;
    bNeurons[1] = 0.2f;
    cNeurons[1] = -65.0f;
    dNeurons[1] = 2.0f;

    // Chattering
    aNeurons[2] = 0.02f;
    bNeurons[2] = 0.2f;
    cNeurons[2] = -50.0f;
    dNeurons[2] = 2.0f;

    // Bursting
    aNeurons[3] = 0.02f;
    bNeurons[3] = 0.2f;
    cNeurons[3] = -55.0f;
    dNeurons[3] = 4.0f;
    
    runtime.initializeSparse();
    
    std::ofstream file("voltages.csv");
    file << "Time(ms), Neuron number, Voltage (mV)" << std::endl;
    
    auto *vNeuronsArray = runtime.getArray(*neurons, "V");
    const float *vNeurons = vNeuronsArray->getHostPointer<float>();
    while(runtime.getTime() < 200.0f) {
        runtime.stepTime();
        vNeuronsArray->pullFromDevice();
        for(unsigned int n = 0; n < 4; n++) {
            file << runtime.getTime() << ", " << n << ", " << vNeurons[n] << std::endl;
        }
    }
    
}
