#include "modelSpec.h"

void modelDefinition(NNmodel &model)
{
  initGeNN();
  model.setDT(1.0);
  model.setName("vogels_2011");
}