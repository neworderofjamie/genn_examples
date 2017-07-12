#include "modelSpec.h"

//----------------------------------------------------------------------------
// IzhikevichVS
//----------------------------------------------------------------------------
class IzhikevichV : public NeuronModels::Izhikevich
{
public:
    DECLARE_MODEL(IzhikevichV, 1, 6);

    SET_SIM_CODE(
        "    if ($(V) >= 30.0){\n"
        "      $(V)=$(c);\n"
        "                  $(U)+=$(d);\n"
        "    } \n"
        "    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset))*DT; //at two times for numerical stability\n"
        "    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset))*DT;\n"
        "    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
        "    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n"
        "   //if ($(V) > 30.0){   //keep this only for visualisation -- not really necessaary otherwise \n"
        "   //  $(V)=30.0; \n"
        "   //}\n"
    );
    
    SET_PARAM_NAMES({"Ioffset"});

    SET_VARS({{"V","scalar"}, {"U", "scalar"}, {"a", "scalar"}, {"b", "scalar"}, {"c", "scalar"}, {"d", "scalar"}});
};
IMPLEMENT_MODEL(IzhikevichV);

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(0.1);
  model.setName("izk_regimes");

  // Izhikevich model parameters
  auto paramValues = IzhikevichV::ParamValues(10.0);
  auto initValues = IzhikevichV::VarValues(-65.0, -20.0, 0.02, 0.2, -65.0, 8.0);

  // Create population of Izhikevich neurons
  model.addNeuronPopulation<IzhikevichV>("Neurons", 4, paramValues, initValues);
  model.finalize();
}