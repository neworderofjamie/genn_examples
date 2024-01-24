
//----------------------------------------------------------------------------
// DVS
//----------------------------------------------------------------------------
class DVS : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(DVS);
    SET_THRESHOLD_CONDITION_CODE("spikeVector[id / 32] & (1 << (id % 32))");
    SET_EXTRA_GLOBAL_PARAMS( {{"spikeVector", "uint32_t*"}} );
};
IMPLEMENT_SNIPPET(DVS);

//----------------------------------------------------------------------------
// OutputClassification
//----------------------------------------------------------------------------
class OutputClassification : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(OutputClassification);

    SET_PARAMS({"TauOut"});    // Membrane time constant [ms]

    SET_VARS({{"Y", "scalar"}, {"B", "scalar", VarAccess::READ_ONLY}});

    SET_DERIVED_PARAMS({
        {"Kappa", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauOut").cast<double>()); }}});

    SET_SIM_CODE(
        "Y = (Kappa * Y) + Isyn + B;\n");
};
IMPLEMENT_SNIPPET(OutputClassification);

//----------------------------------------------------------------------------
// RecurrentALIF
//----------------------------------------------------------------------------
class RecurrentALIF : public NeuronModels::Base
{
public:
    DECLARE_SNIPPET(RecurrentALIF);

    SET_PARAMS({
        "TauM",         // Membrane time constant [ms]
        "TauAdap",      // Adaption time constant [ms]
        "Vthresh",      // Spiking threshold [mV]
        "TauRefrac",    // Refractory time constant [ms]
        "Beta"});       // Scale of adaption [mV]

    SET_VARS({{"V", "scalar"}, {"A", "scalar"}, {"RefracTime", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Alpha", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauM").cast<double>()); }},
        {"Rho", [](const ParamValues &pars, double dt){ return std::exp(-dt / pars.at("TauAdap").cast<double>()); }}});

    SET_ADDITIONAL_INPUT_VARS({{"IsynFeedback", "scalar", 0.0}});

    SET_SIM_CODE(
        "V = (Alpha * V) + Isyn;\n"
        "A *= Rho;\n"
        "if (RefracTime > 0.0) {\n"
        "  RefracTime -= dt;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("RefracTime <= 0.0 && V >= (Vthresh + (Beta * A))");

    SET_RESET_CODE(
        "RefracTime = TauRefrac;\n"
        "V -= Vthresh;\n"
        "A += 1.0;\n");
};
IMPLEMENT_SNIPPET(RecurrentALIF);

void modelDefinition(ModelSpec &model)
{
    model.setDT(1.0);
    model.setName("dvs_classifier");
    //model.setTiming(Parameters::measureTiming);
    model.setMergePostsynapticModels(true);

    //---------------------------------------------------------------------------
    // Build model
    //---------------------------------------------------------------------------
    // LIF model parameters for P population
    ParamValues hiddenParam{
        {"TauM", 20.0},
        {"TauAdap", 2000.0},
        {"Vthresh", 0.6},
        {"TauRefrac", 5.0},
        {"Beta", 0.0174}};

    // LIF initial conditions
    VarValues hiddenInit{
        {"V", 0.0},
        {"A", 0.0},
        {"RefracTime", 0.0}};

    ParamValues outputParam{
        {"TauOut", 20.0}};

    VarValues outputInit{
        {"Y", 0.0},
        {"B", uninitialisedVar()}};

    VarValues weightInit{
        {"g", uninitialisedVar()}};

    //------------------------------------------------------------------------
    // Neuron populations
    //------------------------------------------------------------------------
    auto *dvs = model.addNeuronPopulation<DVS>("DVS", 32 * 32);
    auto *hidden1 = model.addNeuronPopulation<RecurrentALIF>("Hidden1", 256, hiddenParam, hiddenInit);
    auto *hidden2 = model.addNeuronPopulation<RecurrentALIF>("Hidden2", 256, hiddenParam, hiddenInit);
    auto *output = model.addNeuronPopulation<OutputClassification>("Output", 11, outputParam, outputInit);

    //------------------------------------------------------------------------
    // Synapse populations
    //------------------------------------------------------------------------
    model.addSynapsePopulation("DVS_Hidden1", SynapseMatrixType::DENSE,
                               dvs, hidden1,
                               initWeightUpdate<WeightUpdateModels::StaticPulse>({}, weightInit),
                               initPostsynaptic<PostsynapticModels::DeltaCurr>());
    model.addSynapsePopulation("Hidden1_Hidden2", SynapseMatrixType::DENSE,
                               hidden1, hidden2,
                               initWeightUpdate<WeightUpdateModels::StaticPulse>({}, weightInit),
                               initPostsynaptic<PostsynapticModels::DeltaCurr>());
    model.addSynapsePopulation("Hidden2_Hidden2", SynapseMatrixType::DENSE,
                               hidden2, hidden2,
                               initWeightUpdate<WeightUpdateModels::StaticPulse>({}, weightInit),
                               initPostsynaptic<PostsynapticModels::DeltaCurr>());
    model.addSynapsePopulation("Hidden2_Output", SynapseMatrixType::DENSE,
                               hidden2, output,
                               initWeightUpdate<WeightUpdateModels::StaticPulse>({}, weightInit),
                               initPostsynaptic<PostsynapticModels::DeltaCurr>());
}

void simulate(const ModelSpec &model, Runtime::Runtime &runtime)
{
}
