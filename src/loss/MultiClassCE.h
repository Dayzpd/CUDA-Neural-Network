
#ifndef MULTI_CLASS_CE_H
#define MULTI_CLASS_CE_H

#include "LossFunction.h"

namespace neural_network
{

  class MultiClassCE
  {
    public:
      float calculate(Neurons& prediction, Neurons& actual);

      float calculate_deriv(Neurons& prediction, Neurons& actual);
  }

}

#endif
