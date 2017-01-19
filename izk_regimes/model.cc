#include "modelSpec.h"

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(0.1);
  model.setName("izk_regimes");

  // Izhikevich model parameters
  double parameters[1] =
  {
    10.0, // 0 - i offset
  };

  double initialConditions[7] =
  {
    -65.0,  // 0 - V
    -20.0,  // 1 - U
    0.02,   // 2 - a
    0.2,    // 3 - b
    -65.0,  // 4 - c
    8.0,    // 5 - d
  };

  // Copy standard Izhikevich neuron and modify simulation code so offset current is applied
  auto n = nModels[IZHIKEVICH_V];
  n.pNames.push_back("Ioffset");
  n.simCode= "    if ($(V) >= 30.0){\n\
      $(V)=$(c);\n\
      $(U)+=$(d);\n\
    } \n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset))*DT; //at two times for numerical stability\n\
    $(V)+=0.5*(0.04*$(V)*$(V)+5.0*$(V)+140.0-$(U)+$(Isyn)+$(Ioffset))*DT;\n\
    $(U)+=$(a)*($(b)*$(V)-$(U))*DT;\n\
    //if ($(V) > 30.0){      //keep this only for visualisation -- not really necessaary otherwise \n\
    //  $(V)=30.0; \n\
    //}\n";
  const unsigned int MY_IZHIKEVICH_V = nModels.size();
  nModels.push_back(n);

  // Create population of Izhikevich neurons
  model.addNeuronPopulation("Neurons", 4, MY_IZHIKEVICH_V,
                            parameters, initialConditions);
  model.finalize();
}